"""第13阶段第18课：MCP 2026-07-28 生产环境授权。

对当前 MCP 授权面的标准库演练：

  - RFC 8414 授权服务器元数据
  - 客户端 ID 元数据文档优先，已弃用的 RFC 7591 DCR 作为后备
  - PKCE（RFC 7636）带受众绑定的授权码流程（RFC 8707）
  - RFC 9207 授权响应发行方验证
  - 资源服务器上的 JWT 验证
  - JWKS 缓存定时刷新（IdP 轮换密钥；资源服务器仅重新获取它们）
  - 通过 aud 声明拒绝受众重放
  - 客户端注册按发行方索引，访问令牌按发行方加资源索引

系统由三个角色建模：一个签发令牌并轮换签名密钥的 AuthorizationServer，一个缓存 JWKS 并验证每个请求的 ResourceServer（MCP 服务器），以及一个注册并获取令牌的 Client。

仅使用标准库。运行：python3 main.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# JWT 辅助工具——HS256 仅用于保持课程只依赖标准库；生产环境使用 RS256/EdDSA
# ---------------------------------------------------------------------------


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def jwt_sign(payload: dict, kid: str, secret: bytes) -> str:
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    h = b64url(json.dumps(header, separators=(",", ":")).encode())
    p = b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret, f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64url(sig)}"


def jwt_decode(token: str) -> tuple[dict, dict, str]:
    h_b64, p_b64, sig_b64 = token.split(".")
    header = json.loads(b64url_decode(h_b64))
    payload = json.loads(b64url_decode(p_b64))
    return header, payload, sig_b64


def jwt_verify(token: str, secret: bytes) -> bool:
    h_b64, p_b64, sig_b64 = token.split(".")
    expected = hmac.new(secret, f"{h_b64}.{p_b64}".encode(), hashlib.sha256).digest()
    return hmac.compare_digest(expected, b64url_decode(sig_b64))


def protected_resource_metadata_url(resource: str) -> str:
    parsed = urlparse(resource)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("MCP resource must be an absolute HTTPS URL without query or fragment")
    suffix = "" if parsed.path in {"", "/"} else parsed.path
    return f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{suffix}"


MCP_RESOURCE = "https://notes.example.com"
OTHER_MCP_RESOURCE = "https://tasks.example.com"

# RFC 9728 受保护资源元数据 URL。每个 401/403 响应都会在此
# WWW-Authenticate 标头中指明，以便客户端可以重新发现授权服务器。
MCP_RESOURCE_METADATA = protected_resource_metadata_url(MCP_RESOURCE)
OTHER_MCP_RESOURCE_METADATA = protected_resource_metadata_url(OTHER_MCP_RESOURCE)

# 每个工具声明其所需的权限范围。破坏性工具位于更强的
# 权限范围（mcp:tools.delete）之后；该范围不在 IdP 的最小 scopes_supported 中，因此
# 客户端只能通过权限提升流程访问它。
TOOL_SCOPES = {
    "notes.list": "mcp:tools.invoke",
    "notes.read": "mcp:tools.invoke",
    "notes.delete": "mcp:tools.delete",
    "tasks.list": "mcp:tools.invoke",
}
DEFAULT_TOOL_SCOPE = "mcp:tools.invoke"
AUTHORIZATION_CODE_TTL_SECONDS = 300


def parsed_absolute_redirect_uri(value: object):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        return None
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or parsed.fragment or parsed.username is not None or parsed.password is not None:
        return None
    if parsed.scheme in {"http", "https"} and (not parsed.netloc or hostname is None):
        return None
    return parsed


def valid_web_redirect_uri(value: object) -> bool:
    parsed = parsed_absolute_redirect_uri(value)
    return parsed is not None and parsed.scheme == "https" and parsed.hostname is not None


def valid_private_use_scheme(scheme: str) -> bool:
    labels = scheme.split(".")
    return len(labels) >= 2 and all(
        label
        and label.isascii()
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


def valid_native_redirect_uri(value: object) -> bool:
    parsed = parsed_absolute_redirect_uri(value)
    if parsed is None:
        return False
    if parsed.scheme == "https":
        return parsed.hostname is not None
    if parsed.scheme == "http":
        return parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    return valid_private_use_scheme(parsed.scheme)


# ---------------------------------------------------------------------------
# 授权服务器 - 签发令牌、注册客户端、轮换签名密钥
# ---------------------------------------------------------------------------


@dataclass
class IdPKey:
    kid: str
    secret: bytes
    issued_at: float


@dataclass
class AuthorizationServer:
    issuer: str = "https://auth.example.com"
    keys: list[IdPKey] = field(default_factory=list)
    clients: dict[str, dict] = field(default_factory=dict)
    authorization_codes: dict[str, dict] = field(default_factory=dict)
    _authorization_codes_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def current_key(self) -> IdPKey:
        return self.keys[-1]

    def rotate_key(self) -> IdPKey:
        """授权服务器端密钥轮换：引入下一个密钥，淘汰最旧的密钥。

        稳态是两个重叠的密钥，因此由前一个密钥签名的
        令牌在过期前保持有效。
        """
        new_kid = f"k_{int(time.time())}_{secrets.token_hex(2)}"
        new = IdPKey(kid=new_kid, secret=secrets.token_bytes(32), issued_at=time.time())
        self.keys.append(new)
        if len(self.keys) > 2:
            self.keys = self.keys[-2:]
        return new

    def jwks(self) -> dict:
        return {
            "keys": [
                {"kid": k.kid, "kty": "oct", "alg": "HS256", "use": "sig", "k": b64url(k.secret)}
                for k in self.keys
            ]
        }

    def metadata(self) -> dict:
        """RFC 8414授权服务器元数据。"""
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "jwks_uri": f"{self.issuer}/.well-known/jwks.json",
            "registration_endpoint": f"{self.issuer}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
            "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
            "authorization_response_iss_parameter_supported": True,
            "client_id_metadata_document_supported": True,
        }

    def register_cimd(self, document_url: str, document: dict) -> str:
        """解析客户端ID元数据文档，无需生成标识符。"""
        parsed = urlparse(document_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path in {"", "/"}:
            raise ValueError("CIMD client_id must be an absolute HTTPS URL with a path")
        if document.get("client_id") != document_url:
            raise ValueError("CIMD client_id must equal its document URL")
        client_name = document.get("client_name")
        if not isinstance(client_name, str) or not client_name.strip():
            raise ValueError("CIMD requires a non-empty client_name")
        application_type = document.get("application_type")
        if application_type is not None and application_type not in {"native", "web"}:
            raise ValueError("CIMD application_type, when present, must be native or web")
        redirect_application_type = application_type or "native"
        redirect_uris = document.get("redirect_uris", [])
        if (
            not isinstance(redirect_uris, list)
            or not redirect_uris
            or any(parsed_absolute_redirect_uri(uri) is None for uri in redirect_uris)
        ):
            raise ValueError(
                "CIMD requires absolute redirect URIs without fragments"
            )
        if redirect_application_type == "web" and any(
            not valid_web_redirect_uri(uri) for uri in redirect_uris
        ):
            raise ValueError(
                "CIMD web clients require absolute HTTPS redirect URIs "
                "with a host and no fragment"
            )
        if redirect_application_type == "native" and any(
            not valid_native_redirect_uri(uri) for uri in redirect_uris
        ):
            raise ValueError(
                "CIMD native clients require HTTPS, a loopback HTTP URI, or a "
                "domain-based private-use scheme"
            )
        self.clients[document_url] = {
            "redirect_uris": redirect_uris,
            "grant_types": document.get("grant_types", ["authorization_code"]),
            "application_type": application_type,
            "client_name": client_name,
            "enrollment": "cimd",
            "issued_at": time.time(),
        }
        return document_url

    def register_client(self, body: dict) -> dict:
        """已弃用的RFC 7591注册保留以兼容旧版本。"""
        redirect_uris = body.get("redirect_uris", [])
        if (
            not isinstance(redirect_uris, list)
            or not redirect_uris
            or any(parsed_absolute_redirect_uri(uri) is None for uri in redirect_uris)
        ):
            return {"status": 400, "body": {"error": "invalid_redirect_uri"}}
        application_type = body.get("application_type")
        if application_type not in {"native", "web"}:
            return {"status": 400, "body": {"error": "invalid_client_metadata"}}
        if application_type == "web" and any(
            not valid_web_redirect_uri(uri) for uri in redirect_uris
        ):
            return {"status": 400, "body": {"error": "invalid_redirect_uri"}}
        if application_type == "native" and any(
            not valid_native_redirect_uri(uri) for uri in redirect_uris
        ):
            return {"status": 400, "body": {"error": "invalid_redirect_uri"}}
        if body.get("token_endpoint_auth_method", "none") not in {"none", "private_key_jwt"}:
            return {"status": 400, "body": {"error": "invalid_client_metadata"}}
        cid = f"c_{secrets.token_hex(4)}"
        reg_token = secrets.token_urlsafe(24)
        self.clients[cid] = {
            "redirect_uris": redirect_uris,
            "grant_types": body.get("grant_types", ["authorization_code"]),
            # 仅存储哈希；窃取此令牌会让攻击者重写重定向 URI。
            "registration_access_token_hash": hashlib.sha256(reg_token.encode()).hexdigest(),
            "client_name": body.get("client_name", ""),
            "application_type": application_type,
            "enrollment": "dcr",
            "issued_at": time.time(),
        }
        return {
            "status": 201,
            "body": {
                "client_id": cid,
                "client_id_issued_at": int(time.time()),
                "redirect_uris": redirect_uris,
                "grant_types": body.get("grant_types", ["authorization_code"]),
                "application_type": application_type,
                "registration_access_token": reg_token,
                "registration_client_uri": f"{self.issuer}/register/{cid}",
            },
        }

    def pre_register_client(
        self,
        client_id: str,
        *,
        redirect_uris: list[str],
        client_name: str,
        application_type: str = "native",
    ) -> str:
        if not client_id or not redirect_uris or not client_name.strip():
            raise ValueError("pre-registration requires client_id, client_name, and redirect_uris")
        if application_type not in {"native", "web"}:
            raise ValueError("pre-registration application_type must be native or web")
        if (
            not isinstance(redirect_uris, list)
            or any(parsed_absolute_redirect_uri(uri) is None for uri in redirect_uris)
        ):
            raise ValueError(
                "pre-registration requires absolute redirect URIs without fragments"
            )
        if application_type == "web" and any(
            not valid_web_redirect_uri(uri) for uri in redirect_uris
        ):
            raise ValueError(
                "pre-registered web clients require absolute HTTPS redirect URIs "
                "with a host and no fragment"
            )
        if application_type == "native" and any(
            not valid_native_redirect_uri(uri) for uri in redirect_uris
        ):
            raise ValueError(
                "pre-registered native clients require HTTPS redirect URIs, "
                "loopback HTTP redirect URIs, or a domain-based private-use scheme"
            )
        self.clients[client_id] = {
            "redirect_uris": list(redirect_uris),
            "grant_types": ["authorization_code"],
            "application_type": application_type,
            "client_name": client_name,
            "enrollment": "pre_registered",
            "issued_at": time.time(),
        }
        return client_id

    def begin_authorization(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        scopes: set[str],
        resource: str,
        user: str,
    ) -> dict[str, str]:
        client = self.clients.get(client_id)
        if client is None:
            raise ValueError("client is not enrolled with this issuer")
        if redirect_uri not in client.get("redirect_uris", []):
            raise ValueError("authorization redirect_uri is not registered")
        if not isinstance(code_challenge, str) or not code_challenge:
            raise ValueError("authorization request requires an S256 code_challenge")
        if code_challenge_method != "S256":
            raise ValueError("authorization request requires code_challenge_method S256")
        parsed_resource = urlparse(resource)
        if parsed_resource.scheme != "https" or not parsed_resource.netloc:
            raise ValueError("resource must be an absolute HTTPS URL")
        with self._authorization_codes_lock:
            now = time.time()
            expired_codes = [
                code
                for code, record in self.authorization_codes.items()
                if record["expires_at"] <= now
            ]
            for expired_code in expired_codes:
                self.authorization_codes.pop(expired_code, None)
            code = secrets.token_urlsafe(24)
            while code in self.authorization_codes:
                code = secrets.token_urlsafe(24)
            self.authorization_codes[code] = {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method,
                "scopes": set(scopes),
                "resource": resource,
                "user": user,
                "expires_at": now + AUTHORIZATION_CODE_TTL_SECONDS,
            }
        return {"code": code, "iss": self.issuer}

    def redeem_code(
        self,
        *,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
        resource: str,
    ) -> str:
        with self._authorization_codes_lock:
            record = self.authorization_codes.get(code)
            if record is None:
                raise ValueError("authorization code is invalid or already used")
            if record["expires_at"] <= time.time():
                self.authorization_codes.pop(code, None)
                raise ValueError("authorization code is expired")
            if record["client_id"] != client_id or record["redirect_uri"] != redirect_uri:
                raise ValueError("authorization code is not bound to this client redirect")
            if record["resource"] != resource:
                raise ValueError("token resource does not match the authorization request")
            supplied_challenge = b64url(hashlib.sha256(code_verifier.encode()).digest())
            if not hmac.compare_digest(record["code_challenge"], supplied_challenge):
                raise ValueError("PKCE code_verifier does not match the stored challenge")
            self.authorization_codes.pop(code)
        return self.issue_token(
            client_id,
            record["user"],
            record["scopes"],
            record["resource"],
        )

    def issue_token(self, client_id: str, user: str, scopes: set[str], resource: str) -> str:
        """签发由当前密钥签名且绑定受众的访问令牌。"""
        if client_id not in self.clients:
            raise ValueError("client is not enrolled with this issuer")
        key = self.current_key()
        claims = {
            "iss": self.issuer,
            "sub": user,
            "aud": resource,
            "azp": client_id,
            "scope": " ".join(sorted(scopes)),
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        return jwt_sign(claims, kid=key.kid, secret=key.secret)


# ---------------------------------------------------------------------------
# 资源服务器（MCP 服务器）——缓存 JWKS，验证每个请求
# ---------------------------------------------------------------------------


@dataclass
class ResourceServer:
    resource: str
    auth_server: AuthorizationServer
    allowed_issuers: list[str] = field(default_factory=list)
    jwks_cache: dict[str, dict] = field(default_factory=dict)

    @property
    def resource_metadata(self) -> str:
        return protected_resource_metadata_url(self.resource)

    def refresh_jwks(self) -> dict:
        """将授权服务器发布的 JWKS 重新获取到缓存中。幂等。

        密钥*轮换*发生在授权服务器上，而非此处。资源
        服务器无法生成或滚动授权服务器的签名密钥；它只能重新拉取已发布的
        密钥集。定时刷新任务和验证器的
        缓存未命中后备流程都会调用此方法。因为这是一个纯粹的获取操作，攻击者
        发送带有随机`kid`值的令牌最多只会触发一次无害的
        重新获取，而非无限系列的密钥轮换（如果将后备流程连接到
        密钥轮换与令牌签发操作，就会出现这类缺陷）。
        """
        keys = self.auth_server.jwks()["keys"]
        self.jwks_cache[self.auth_server.issuer] = {"keys": keys, "fetched_at": time.time()}
        return {"refreshed": True, "kids": [k["kid"] for k in keys]}

    def cached_kids(self) -> list[str]:
        entry = self.jwks_cache.get(self.auth_server.issuer, {"keys": []})
        return [k["kid"] for k in entry["keys"]]

    def validate(self, token: str, required_scope: str | None = None) -> dict:
        rm = self.resource_metadata

        def challenge(status: int, params: str) -> dict:
            return {"valid": False, "status": status, "www_authenticate": f"Bearer {params}"}

        try:
            header, claims, _ = jwt_decode(token)
        except Exception:
            return challenge(401, f'error="invalid_token", error_description="malformed", resource_metadata="{rm}"')

        iss = claims.get("iss", "")
        # 首先检查发行方允许列表：不受信任的 iss 不应让我们付出
        # 一次 JWKS 刷新的代价，且 "iss not allowed" 是应返回的正确错误。
        if iss not in self.allowed_issuers:
            return challenge(401, f'error="invalid_token", error_description="iss not allowed", resource_metadata="{rm}"')
        cache = self.jwks_cache.get(iss)
        if cache is None:
            self.refresh_jwks()
            cache = self.jwks_cache.get(iss)

        matching = next((k for k in cache["keys"] if k["kid"] == header.get("kid")), None) if cache else None
        if matching is None:
            # 密钥重叠窗口：由比我们缓存更新的密钥签名的令牌。
            # 重新获取（而非轮换）一次，然后重新检查。伪造的 kid 只需
            # 在一次幂等获取后直接落入下面的401。
            self.refresh_jwks()
            cache = self.jwks_cache.get(iss)
            matching = next((k for k in cache["keys"] if k["kid"] == header.get("kid")), None) if cache else None
        if matching is None:
            return challenge(401, f'error="invalid_token", error_description="unknown kid", resource_metadata="{rm}"')

        if not jwt_verify(token, b64url_decode(matching["k"])):
            return challenge(401, f'error="invalid_token", error_description="bad signature", resource_metadata="{rm}"')
        if claims.get("aud") != self.resource:
            return challenge(401, f'error="invalid_token", error_description="audience mismatch", resource_metadata="{rm}"')
        if claims.get("exp", 0) < time.time():
            return challenge(401, f'error="invalid_token", error_description="expired", resource_metadata="{rm}"')
        if required_scope and required_scope not in set(claims.get("scope", "").split()):
            return challenge(403, f'error="insufficient_scope", scope="{required_scope}", resource_metadata="{rm}"')
        return {"valid": True, "claims": claims}

    def call_tool(self, tool: str, bearer: str) -> dict:
        required_scope = TOOL_SCOPES.get(tool, DEFAULT_TOOL_SCOPE)
        result = self.validate(bearer, required_scope=required_scope)
        if not result["valid"]:
            return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
        return {"status": 200, "body": {"tool": tool, "user": result["claims"]["sub"], "ok": True}}


# ---------------------------------------------------------------------------
# 客户端——发现、DCR 注册、PKCE 和受众绑定令牌请求
# ---------------------------------------------------------------------------


@dataclass
class Client:
    name: str
    auth_server: AuthorizationServer
    client_metadata_url: str | None = None
    client_metadata: dict | None = None
    pre_registered_client_ids_by_issuer: dict[str, str] = field(default_factory=dict)
    client_ids_by_issuer: dict[str, str] = field(default_factory=dict)
    access_tokens_by_issuer_resource: dict[tuple[str, str], str] = field(default_factory=dict)
    expected_issuer: str | None = None
    require_response_issuer: bool = False

    def discover(self) -> dict:
        meta = self.auth_server.metadata()
        if meta.get("issuer") != self.auth_server.issuer:
            raise ValueError("authorization metadata issuer mismatch")
        if "S256" not in meta["code_challenge_methods_supported"]:
            raise ValueError("authorization server does not advertise S256 PKCE")
        if not (meta.get("client_id_metadata_document_supported") or "registration_endpoint" in meta):
            raise ValueError("authorization server advertises no client enrollment path")
        self.expected_issuer = meta["issuer"]
        self.require_response_issuer = bool(
            meta.get("authorization_response_iss_parameter_supported")
        )
        return meta

    def register(self) -> str:
        """使用已弃用的DCR后备，并按发行方索引凭据。"""
        resp = self.auth_server.register_client(
            {
                "redirect_uris": ["http://127.0.0.1:7333/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
                "application_type": "native",
                "scope": "mcp:tools.invoke",
                "client_name": self.name,
            }
        )
        if resp["status"] != 201:
            raise ValueError(f"client registration failed: {resp}")
        issuer = self.auth_server.issuer
        self.client_ids_by_issuer[issuer] = resp["body"]["client_id"]
        return self.client_ids_by_issuer[issuer]

    def enroll(self) -> str:
        """优先使用CIMD；仅在当前发行方无法解析时使用DCR。"""
        meta = self.discover()
        issuer = meta["issuer"]
        if issuer in self.client_ids_by_issuer:
            return self.client_ids_by_issuer[issuer]
        pre_registered = self.pre_registered_client_ids_by_issuer.get(issuer)
        if pre_registered is not None:
            if pre_registered not in self.auth_server.clients:
                raise ValueError("pre-registered client_id is not known to this issuer")
            self.client_ids_by_issuer[issuer] = pre_registered
            return pre_registered
        if meta.get("client_id_metadata_document_supported"):
            if not self.client_metadata_url or not self.client_metadata:
                raise ValueError("CIMD-capable issuer requires a client metadata document")
            client_id = self.auth_server.register_cimd(
                self.client_metadata_url, self.client_metadata
            )
            self.client_ids_by_issuer[issuer] = client_id
            return client_id
        return self.register()

    def validate_authorization_response_issuer(self, returned_issuer: str | None) -> None:
        if self.expected_issuer is None:
            raise ValueError("authorization server metadata was not discovered")
        if returned_issuer is None:
            if self.require_response_issuer:
                raise ValueError("authorization response omitted required iss")
            return
        if returned_issuer != self.expected_issuer:
            raise ValueError("authorization response issuer mismatch")

    def use_authorization_server(self, auth_server: AuthorizationServer) -> None:
        """切换发行方，无需复制客户端标识符或访问令牌。"""
        self.auth_server = auth_server
        self.expected_issuer = None
        self.require_response_issuer = False

    def authorize(self, scopes: set[str], resource: str, user: str) -> str:
        issuer = self.auth_server.issuer
        client_id = self.client_ids_by_issuer.get(issuer)
        if client_id is None:
            raise ValueError("client must enroll separately with this issuer")
        redirect_uris = self.auth_server.clients[client_id].get("redirect_uris", [])
        if not redirect_uris:
            raise ValueError("client has no registered redirect URI")
        redirect_uri = redirect_uris[0]
        verifier = secrets.token_urlsafe(32)
        challenge = b64url(hashlib.sha256(verifier.encode()).digest())
        authorization_response = self.auth_server.begin_authorization(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=challenge,
            code_challenge_method="S256",
            scopes=scopes,
            resource=resource,
            user=user,
        )
        self.validate_authorization_response_issuer(authorization_response.get("iss"))
        token = self.auth_server.redeem_code(
            code=authorization_response["code"],
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
            resource=resource,
        )
        self.access_tokens_by_issuer_resource[(issuer, resource)] = token
        return token


# ---------------------------------------------------------------------------
# 演示 - 生产流程
# ---------------------------------------------------------------------------


def demo() -> None:
    print("=" * 72)
    print("第 13 阶段第 18 课 - 生产环境中的 MCP 身份验证")
    print("=" * 72)

    print("\n--- 步骤1：启动授权服务器（两个重叠密钥） ---")
    auth = AuthorizationServer()
    auth.rotate_key()
    auth.rotate_key()
    print(f"  发行方={auth.issuer}, 密钥={[k.kid for k in auth.keys]}")

    print("\n--- 步骤2：客户端发现授权服务器（RFC 8414） ---")
    cimd_url = "https://client.example.com/oauth/client.json"
    client = Client(
        name="原生客户端示例",
        auth_server=auth,
        client_metadata_url=cimd_url,
        client_metadata={
            "client_id": cimd_url,
            "client_name": "原生客户端示例",
            "redirect_uris": ["http://127.0.0.1:7333/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    meta = client.discover()
    print(f"  发行方={meta['issuer']}，支持 S256 PKCE")
    print(f"  支持的 CIMD={meta['client_id_metadata_document_supported']}")

    print("\n--- 步骤3：客户端通过CIMD注册，无需DCR ---")
    cid = client.enroll()
    print(f"  client_id 元数据 URL：{cid}")
    print(f"  凭据缓存发行方密钥: {list(client.client_ids_by_issuer)}")

    print("\n--- 步骤4：客户端运行带资源指示器的PKCE授权流程 ---")
    bearer = client.authorize(scopes={"mcp:tools.invoke"}, resource=MCP_RESOURCE, user="alice@example.com")
    print(f"  已签发 bearer（kid={auth.current_key().kid}，aud={MCP_RESOURCE}）")

    print("\n--- 步骤5：MCP服务器验证请求，JWKS缓存在首次使用时预热 ---")
    server = ResourceServer(resource=MCP_RESOURCE, auth_server=auth, allowed_issuers=[auth.issuer])
    resp = server.call_tool("notes.list", bearer)
    print(f"  服务器响应: {resp}")
    assert resp["status"] == 200

    print("\n--- 步骤6：IdP 轮换密钥，定时刷新重新拉取 JWKS ---")
    print(f"  刷新前缓存的 kid：{server.cached_kids()}")
    auth.rotate_key()  # 授权服务器端轮换，与 MCP 服务器无关
    server.refresh_jwks()  # 定时任务重新拉取已发布的 JWKS
    print(f"  刷新后缓存的 kid：{server.cached_kids()}")

    print("\n--- 步骤7：现有令牌仍然有效（重叠窗口） ---")
    resp = server.call_tool("notes.list", bearer)
    print(f"  服务器响应: {resp}")
    assert resp["status"] == 200

    print("\n--- 步骤8：用新密钥签名的新令牌通过刷新后的 JWKS 验证 ---")
    fresh_bearer = client.authorize(scopes={"mcp:tools.invoke"}, resource=MCP_RESOURCE, user="alice@example.com")
    fresh_header, _, _ = jwt_decode(fresh_bearer)
    print(f"  新令牌 kid：{fresh_header['kid']}")
    resp = server.call_tool("notes.read", fresh_bearer)
    print(f"  服务器响应: {resp}")
    assert resp["status"] == 200

    print("\n--- 步骤9：针对不同 MCP 资源的受众重放尝试 ---")
    other_server = ResourceServer(resource=OTHER_MCP_RESOURCE, auth_server=auth, allowed_issuers=[auth.issuer])
    resp = other_server.call_tool("tasks.list", bearer)
    print(f"  另一服务器响应: {resp}")
    assert resp["status"] == 401
    assert "audience mismatch" in resp["WWW-Authenticate"]

    print("\n--- 附加：用于更高权限范围的权限提升流程 ---")
    elevated = client.authorize(
        scopes={"mcp:tools.invoke", "mcp:tools.delete"}, resource=MCP_RESOURCE, user="alice@example.com"
    )
    elevated_resp = server.call_tool("notes.delete", elevated)
    print(f"  服务器响应: {elevated_resp}")

    print("\n" + "=" * 72)
    print("完成 - 发行方绑定注册、响应 iss、受众和 JWKS 刷新")
    print("=" * 72)


if __name__ == "__main__":
    demo()

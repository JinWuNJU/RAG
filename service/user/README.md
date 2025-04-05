### 用户系统
使用jwt，payload 为`User.id` (uuid4)

### [auth.py](./auth.py)
向其它模块封装了认证功能

#### 方法：`decode_jwt_to_uid`
**功能说明**  
解码 JWT 令牌并提取用户 ID (`User.id`)，验证令牌的有效性。如果令牌无效或过期，则抛出 `HTTP 401 Unauthorized` 异常。

**参数**  
- `Authorize: AuthJWT`  
  依赖注入的 `AuthJWT` 对象，用于处理 JWT 相关操作。

**返回值**  
- 返回解析后的用户 ID，类型为 `UUID`。

**异常处理**  
- 如果 JWT 令牌无效、过期或解析失败，将抛出 `HTTPException`，状态码为 `401 Unauthorized`，错误信息为 `"登陆状态失效"`。

---

#### 示例用例

```python
@router.get("/test")
def test_jwt(Authorize: AuthJWT = Depends()):
    """
    测试 JWT 解码功能
    """
    # 调用 decode_jwt_to_uid 方法验证 JWT 并获取用户 ID
    user_id = decode_jwt_to_uid(Authorize)
    pass
```

**运行结果**  
1. 如果 JWT 令牌无效或缺失：
   - 返回 HTTP 401 错误，响应内容为：`{"detail": "登陆状态失效"}`。

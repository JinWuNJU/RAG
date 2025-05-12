import asyncio
import time
import uuid
from typing import List
from uuid import UUID

from sse_starlette import EventSourceResponse

from rest_model.chat.completions import MessagePayload
from rest_model.chat.history import ChatDetail, ChatDialog, ChatHistory, ChatMessage, ChatMessagePart, ChatTextPart, \
    ChatToolCallPart, ChatToolReturnPart
from rest_model.chat.sse import ChatBeginEvent, ChatEndEvent, ChatEvent, ChatTitleEvent, SseEventPackage, \
    ToolCallEvent, ToolReturnEvent
from rest_model.chat.toolcalls import RetrieveToolReturn, RetrievedDocument, RetrieveParams
from service.ai.chat.service_base import BaseChatService

# Mock数据说明：
# 以下mock数据用于模拟一个物理问答系统的典型交互流程，包含：
# 1. 用户查询：关于量子谐振子的物理问题
# 2. 工具调用：模拟知识库检索过程
# 3. 回答内容：包含公式和解释的完整回答
# 4. 对话历史：模拟多轮对话场景

# 模拟用户查询
mock_query = r"举出量子力学中一维无限深势阱中粒子的能量本征值和波函数问题"

# 模拟AI回答 - 包含公式和解释的完整回答
mock_answer = r'''
以下以“量子力学中一维无限深势阱中粒子的能量本征值和波函数问题”为例，这是前沿物理教材中较为经典且复杂的内容：

**问题描述**： 考虑一个质量为 $m$ 的粒子处于一维无限深势阱中，势阱的范围是 $0 < x < a$，势函数为
\[
V(x) = \begin{cases}
0, & 0 < x < a \\
\infty, & x \leq 0, x \geq a
\end{cases}
\]
求粒子的能量本征值和对应的波函数。

**公式解答**：
根据定态薛定谔方程：
\[
-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2} + V(x)\psi(x) = E\psi(x)
\]
在势阱外（$x \leq 0$ 或 $x \geq a$），由于 $V(x)=\infty$，要使方程有解，只能 $\psi(x)=0$。
在势阱内（$0 < x < a$），$V(x) = 0$，薛定谔方程变为：
\[
-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2} = E\psi(x)
\]
令 $k^2 = \frac{2mE}{\hbar^2}$，则方程变为：
\[
\frac{d^2\psi(x)}{dx^2} + k^2\psi(x) = 0
\]
其通解为：
\[
\psi(x) = A\sin(kx) + B\cos(kx)
\]
根据边界条件 $\psi(0) = 0$，可得：
\[
\psi(0) = A\sin(0) + B\cos(0) = B = 0
\]
再根据边界条件 $\psi(a) = 0$，可得：
\[
\psi(a) = A\sin(ka) = 0
\]
因为 $A \neq 0$（否则波函数恒为零），所以 $ka = n\pi$，$n = 1, 2, 3, \cdots$，即 $k = \frac{n\pi}{a}$。
由 $k^2 = \frac{2mE}{\hbar^2}$，可得能量本征值：
\[
E_n = \frac{n^2\pi^2\hbar^2}{2ma^2}, \quad n = 1, 2, 3, \cdots
\]
对应的波函数为：
\[
\psi_n(x) = \sqrt{\frac{2}{a}}\sin\left(\frac{n\pi x}{a}\right), \quad n = 1, 2, 3, \cdots
\]
这里 $\sqrt{\frac{2}{a}}$ 是归一化常数，通过 $\int_{0}^{a}|\psi_n(x)|^2dx = 1$ 得到。

**Python 计算（以计算前几个能量本征值和绘制前几个波函数为例）**：
```python
import numpy as np
import matplotlib.pyplot as plt

# 常数
hbar = 1.054571817e-34  # 约化普朗克常数，单位 J·s
m = 1.674927498e-27  # 粒子质量，这里假设为质子质量，单位 kg
a = 1e-10  # 势阱宽度，单位 m

# 计算能量本征值
def energy(n):
    return (n ** 2) * (np.pi ** 2) * (hbar ** 2) / (2 * m * (a ** 2))

# 计算波函数
def wave_function(n, x):
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)

# 计算前 5 个能量本征值
n_values = np.arange(1, 6)
energies = [energy(n) for n in n_values]
print("能量本征值：", energies)

# 绘制前 3 个波函数
x = np.linspace(0, a, 1000)
for n in range(1, 4):
    psi = wave_function(n, x)
    plt.plot(x, psi, label=f"n = {n}")

plt.xlabel('x')
plt.ylabel('波函数 $\psi(x)$')
plt.title('一维无限深势阱波函数')
plt.legend()
plt.show()
```

上述代码首先定义了相关的物理常数，然后定义了计算能量本征值和波函数的函数。接着计算了前 5 个能量本征值并打印出来，最后绘制了前 3 个波函数的图像。 
'''


# 工具调用过程 - 包含检索和文档返回两个阶段
mock_tooluse: List[ChatMessagePart] = [
    ChatToolCallPart(
        tool_name="retrieve",
        args=RetrieveParams(
            knowledge_base="物理知识库",
            keyword="定态薛定谔方程"
        ).model_dump()
    ),
    ChatToolReturnPart(
        tool_name="retrieve",
        content=RetrieveToolReturn(
            count=10,
            documents=[
                RetrievedDocument(
                    snippet="定态薛定谔方程为...",
                    url="http://example.com"
                ) for _ in range(10)
            ]
        ).model_dump()
    )
]

mock_history_counter = 0
# 模拟对话历史 - 包含多轮对话和追问场景
mock_history: List[ChatDetail] = [
    ChatDetail(
        id=uuid.uuid4(),
        title=f"{(mock_history_counter:= mock_history_counter + 1)} 定态薛定谔方程和量子谐振子",
        chat=ChatDialog(
            messages= [
                # 初始对话
                ChatMessage(
                    parentId=None,
                    id=(root_id := uuid.uuid4()),
                    role="user",
                    part=[
                        ChatTextPart(content=mock_query),
                    ],
                    timestamp=174257095
                ),
                ChatMessage(
                    parentId=root_id,
                    id=(answer_id_1 := uuid.uuid4()),
                    role="assistant",
                    part=[
                        *mock_tooluse,
                        ChatTextPart(content=mock_answer)
                    ],
                    timestamp=174257099
                ),
                # 第二轮对话
                ChatMessage(
                    parentId=None,
                    id=(root_id_2 := uuid.uuid4()),
                    role="user",
                    part=[
                        ChatTextPart(content=mock_query + " （编辑后的版本二查询）"),
                    ],
                    timestamp=174257120
                ),
                ChatMessage(
                    parentId=root_id_2,
                    id=uuid.uuid4(),
                    role="assistant",
                    part=[
                        *mock_tooluse,
                        ChatTextPart(content=mock_answer + " （编辑后的版本二回答）"),
                    ],
                    timestamp=174257130
                )
            ]
        ),
        updated_at=1742570997,
        created_at=1742570958
    ) for _ in range(6)
]


# 模拟SSE事件流
async def event_generator(chat_id: UUID, user_message_id: UUID, assistant_message_id: UUID):
    # 先发送工具调用事件
    yield SseEventPackage(
        ChatBeginEvent(
            chat_id=chat_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id
        )
    )
    yield SseEventPackage(
        ToolCallEvent(
            data=ChatToolCallPart(
                **mock_tooluse[0].model_dump()
            )
        )
    )
    await asyncio.sleep(.5) # 模拟延迟
    yield SseEventPackage(
        ToolReturnEvent(
            data=ChatToolReturnPart(
                **mock_tooluse[1].model_dump()
            )
        )
    )
    # 分段发送回答内容
    for i in range(0, len(mock_answer), 8):
        chunk = mock_answer[i:i+8]
        yield SseEventPackage(
            ChatEvent(
                content=chunk
            )
        )
        await asyncio.sleep(.05)
        
    yield SseEventPackage(
        ChatEndEvent()
    )
    yield SseEventPackage(
        ChatTitleEvent(
            title=mock_history[0].title
        )
    )

class MockChatService(BaseChatService):
    """聊天服务Mock实现"""
    
    async def delete_chat(self, user_id: uuid.UUID, chat_id: str) -> bool:
        """删除对话"""
        for history in mock_history:
            if str(history.id) == chat_id:
                mock_history.remove(history)
                return True
        return False

    async def get_chat(self, user_id:uuid.UUID, chat_id: str):
        chat = [history for history in mock_history if str(history.id) == chat_id]
        if not chat:
            return {}
        return chat[0]

    async def get_history(self, user_id: uuid.UUID, page: int = 1):
        page_size = 3  # 让分页更加明显
        start = (page - 1) * page_size
        end = start + page_size
        return [ChatHistory(**v.model_dump()) for v in mock_history[start:end]]

    async def message_stream(self, user_id: uuid.UUID, payload: MessagePayload) -> EventSourceResponse:
        current_timestamp = int(time.time())
        new_chat = True
        new_user_message = ChatMessage(
                parentId=None,
                id=(user_message_id := uuid.uuid4()),
                role="user",
                part=[
                    ChatTextPart(content=payload.content),
                ],
                timestamp=current_timestamp
            )

        new_assistant_message = ChatMessage(
            parentId=user_message_id,
            id=(assistant_message_id := uuid.uuid4()),
            role="assistant",
            part=[
                *mock_tooluse,
                ChatTextPart(content=mock_answer),
            ],
            timestamp=current_timestamp
        )
        
        chat_id = uuid.uuid4()
        for history in mock_history:
            if payload.chatId and str(history.id) == payload.chatId:
                for chat in history.chat.messages:
                    if str(chat.id) == payload.parentId:
                        new_user_message.parentId = chat.id
                        break
                new_chat = False
                chat_id = history.id
                history.chat.messages = history.chat.messages + [new_user_message]
                history.updated_at = current_timestamp
                history_item = history

        if new_chat:
            history_item = ChatDetail(
                id=chat_id,
                title="定态薛定谔方程和量子谐振子",
                chat=ChatDialog(messages=[new_user_message]),
                updated_at=current_timestamp,
                created_at=current_timestamp
            )
            mock_history.insert(0, history_item)
        async def update_chat_history():
            nonlocal history_item
            nonlocal new_assistant_message
            await asyncio.sleep(14)
            history_item.chat.messages = history_item.chat.messages + [new_assistant_message]

        asyncio.create_task(update_chat_history())
        return EventSourceResponse(event_generator(chat_id, user_message_id, assistant_message_id))
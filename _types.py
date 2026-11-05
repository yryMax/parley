import typing as t
from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Feedback(BaseModel):
    prompt: str
    improvement: str


class TreeNode(BaseModel):
    children: t.List["TreeNode"] # 子节点
    conversation: t.List[Message] # 和攻击模型完整的对话历史
    feedback: t.Optional[Feedback] # 攻击模型的输出，payload + 比上一轮的改进
    response: t.Optional[str] # 目标模型对当前prompt的回复
    on_topic: t.Optional[bool] # evaluator 判断当前prompt是否与目标任务相关
    score: t.Optional[int] # evaluator 对当前prompt的评分


class Parameters(BaseModel):
    model: str
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.9


ChatFunction = t.Callable[[t.List[Message]], Message]
Conversation = t.List[Message]

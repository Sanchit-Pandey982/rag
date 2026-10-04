from pydantic import BaseModel


class UsageDayResponse(BaseModel):
    date: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    requests: int


class UsageTotalResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    requests: int


class UsageSummaryResponse(BaseModel):
    user_id: str
    days: int
    total: UsageTotalResponse
    daily: list[UsageDayResponse]

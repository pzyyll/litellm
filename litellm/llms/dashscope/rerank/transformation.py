import uuid
import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.llms.cohere.common_utils import CohereError
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankResponse,
    RerankResponseMeta,
)
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any


class DashscopeRerankRequestParams(BaseModel):
    top_n: Optional[int] = None
    return_documents: Optional[bool] = None


class DashscopeRerankRequestInput(BaseModel):
    query: str
    documents: List[Union[str, dict]]


class DashscopeRerankRequest(BaseModel):
    model: str
    input: DashscopeRerankRequestInput
    parameters: Optional[DashscopeRerankRequestParams] = None


class DashscopeRerankConfig(BaseRerankConfig):
    """
    Reference: https://help.aliyun.com/zh/model-studio/developer-reference/text-rerank-api
    """

    DEFAULT_BASE_URL = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if not api_key:
            api_key = get_secret_str("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("API key is required. Please set 'DASHSCOPE_API_KEY'")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        default_headers.update(headers)
        return default_headers

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base:
            api_base = api_base.rstrip("/")
            if api_base.endswith("/api/v1"):
                api_base = f"{api_base}/services/rerank/text-rerank/text-rerank"
            return api_base
        return self.DEFAULT_BASE_URL

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> OptionalRerankParams:
        return OptionalRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
        )

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: OptionalRerankParams,
        headers: dict,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Dashscope rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Dashscope rerank")

        parameters = DashscopeRerankRequestParams(
            top_n=optional_rerank_params.get("top_n", None),
            return_documents=optional_rerank_params.get("return_documents", None),
        )

        request = DashscopeRerankRequest(
            model=model,
            input=DashscopeRerankRequestInput(
                query=optional_rerank_params["query"],
                documents=optional_rerank_params["documents"],
            ),
            parameters=parameters,
        )

        return request.model_dump(exclude_none=True)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise CohereError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        if "output" not in raw_response_json:
            raise CohereError(
                message=raw_response.text,
                status_code=raw_response.status_code,
            )

        total_tokens = raw_response_json.get("usage", {}).get("total_tokens", 0)
        meta = RerankResponseMeta(
            billed_units={"total_tokens": total_tokens},
        )

        response = RerankResponse(
            id=raw_response_json.get("request_id") or str(uuid.uuid4()),
            results=raw_response_json.get("output", {}).get("results"),
            meta=meta,
        )

        return response

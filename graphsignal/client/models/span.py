# coding: utf-8

"""
    Graphsignal API

    API for uploading and querying spans, scores, metrics, and logs.

    The version of the OpenAPI document: 1.0.0
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional
from graphsignal.client.models.config_entry import ConfigEntry
from graphsignal.client.models.exception import Exception
from graphsignal.client.models.payload import Payload
from graphsignal.client.models.tag import Tag
from graphsignal.client.models.usage_counter import UsageCounter
from typing import Optional, Set
from typing_extensions import Self

class Span(BaseModel):
    """
    Span
    """ # noqa: E501
    span_id: StrictStr = Field(description="Unique identifier for the span.")
    root_span_id: Optional[StrictStr] = Field(default=None, description="Identifier of the root span, if this is a child span.")
    parent_span_id: Optional[StrictStr] = Field(default=None, description="Identifier of the parent span, if this is a nested span.")
    start_us: StrictInt = Field(description="Start time of the span in microseconds.")
    end_us: StrictInt = Field(description="End time of the span in microseconds.")
    latency_ns: Optional[StrictInt] = Field(default=None, description="Latency in nanoseconds, if applicable.")
    ttft_ns: Optional[StrictInt] = Field(default=None, description="Time to first byte in nanoseconds, if applicable.")
    tags: Optional[List[Tag]] = Field(default=None, description="List of tags associated with the span.")
    exceptions: Optional[List[Exception]] = Field(default=None, description="List of exceptions occurred during the span.")
    payloads: Optional[List[Payload]] = Field(default=None, description="List of payloads related to the span.")
    usage: Optional[List[UsageCounter]] = Field(default=None, description="Usage metrics associated with the span.")
    config: Optional[List[ConfigEntry]] = Field(default=None, description="Configuration entries relevant to the span.")
    __properties: ClassVar[List[str]] = ["span_id", "root_span_id", "parent_span_id", "start_us", "end_us", "latency_ns", "ttft_ns", "tags", "exceptions", "payloads", "usage", "config"]

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.model_dump(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        # TODO: pydantic v2: use .model_dump_json(by_alias=True, exclude_unset=True) instead
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Optional[Self]:
        """Create an instance of Span from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields: Set[str] = set([
        ])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        # override the default output from pydantic by calling `to_dict()` of each item in tags (list)
        _items = []
        if self.tags:
            for _item in self.tags:
                if _item:
                    _items.append(_item.to_dict())
            _dict['tags'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in exceptions (list)
        _items = []
        if self.exceptions:
            for _item in self.exceptions:
                if _item:
                    _items.append(_item.to_dict())
            _dict['exceptions'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in payloads (list)
        _items = []
        if self.payloads:
            for _item in self.payloads:
                if _item:
                    _items.append(_item.to_dict())
            _dict['payloads'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in usage (list)
        _items = []
        if self.usage:
            for _item in self.usage:
                if _item:
                    _items.append(_item.to_dict())
            _dict['usage'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in config (list)
        _items = []
        if self.config:
            for _item in self.config:
                if _item:
                    _items.append(_item.to_dict())
            _dict['config'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of Span from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "span_id": obj.get("span_id"),
            "root_span_id": obj.get("root_span_id"),
            "parent_span_id": obj.get("parent_span_id"),
            "start_us": obj.get("start_us"),
            "end_us": obj.get("end_us"),
            "latency_ns": obj.get("latency_ns"),
            "ttft_ns": obj.get("ttft_ns"),
            "tags": [Tag.from_dict(_item) for _item in obj["tags"]] if obj.get("tags") is not None else None,
            "exceptions": [Exception.from_dict(_item) for _item in obj["exceptions"]] if obj.get("exceptions") is not None else None,
            "payloads": [Payload.from_dict(_item) for _item in obj["payloads"]] if obj.get("payloads") is not None else None,
            "usage": [UsageCounter.from_dict(_item) for _item in obj["usage"]] if obj.get("usage") is not None else None,
            "config": [ConfigEntry.from_dict(_item) for _item in obj["config"]] if obj.get("config") is not None else None
        })
        return _obj


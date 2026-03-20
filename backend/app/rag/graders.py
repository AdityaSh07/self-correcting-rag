from .llm import llm
from .schemas import RelevanceDecision, RetrieveDecision, IsSUPDecision, IsUSEDecision, RewriteDecision

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)
relevance_llm = llm.with_structured_output(RelevanceDecision)
issup_llm = llm.with_structured_output(IsSUPDecision)
isuse_llm = llm.with_structured_output(IsUSEDecision)
rewrite_llm = llm.with_structured_output(RewriteDecision)
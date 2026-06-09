# Stage 5 Trace Flow

Expected request flow for the distributed A2A demo:

1. `test_client.py` creates `trace_id` and `context_id`
2. `customer_agent` receives the user message on port `10100`
3. `customer_agent` discovers `law_agent` through `registry`
4. `law_agent` receives the delegated message on port `10101`
5. `law_agent` decides whether to discover `tax_agent` and `compliance_agent`
6. `law_agent` delegates to `tax_agent` on `10102` when tax issues are present
7. `law_agent` delegates to `compliance_agent` on `10103` when regulatory issues are present
8. `law_agent` aggregates the specialist results
9. `customer_agent` returns the final answer to `test_client.py`

The same `trace_id` should appear in logs for all involved services.

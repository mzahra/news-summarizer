# Lab Summary

The main challenge was integrating Cohere because its API and response format are different from OpenAI and Anthropic. I solved this by reading the documentation and implementing a separate `ask_cohere()` function that converts the response into the same format used by the rest of the pipeline. This lab taught me that supporting multiple LLM providers mainly comes down to handling API differences and normalizing their outputs. As an improvement, I would add a simple integration test to detect SDK or response format changes early.

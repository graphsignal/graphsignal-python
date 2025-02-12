# Graphsignal: LLM Performance Analytics

[![License](http://img.shields.io/github/license/graphsignal/graphsignal-python)](https://github.com/graphsignal/graphsignal-python/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal-python?label=version)](https://github.com/graphsignal/graphsignal-python)


Graphsignal is an observability platform for AI applications. It helps developers ensure AI applications run efficiently and provide the best user experience. With Graphsignal, developers can:

* Measure and optimize LLM performance for APIs and hosted models.
* Analyze model API costs for deployments, models, or any custom tags.
* View and search LLM generations with full context.
* Detect errors and monitor APIs, compute, and GPU utilization.


[![Dashboards](https://graphsignal.com/external/screenshot-dashboard.png)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).


## Install

Install Graphsignal library.

```bash
pip install --upgrade graphsignal
```


## Configure

Configure Graphsignal tracer by specifying your API key directly or via `GRAPHSIGNAL_API_KEY` environment variable.

```python
import graphsignal

graphsignal.configure(api_key='my-api-key', deployment='my-app')
```

To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api-keys) page.

Alternatively, you can add Graphsignal tracer at command line, when running your module or script. Environment variables `GRAPHSIGNAL_API_KEY` and `GRAPHSIGNAL_DEPLOYMENT` must be set.

```bash
python -m graphsignal <script>
```

```bash
python -m graphsignal -m <module>
```


## Integrate

Graphsignal **auto-instruments** and traces libraries and frameworks, such as [OpenAI](https://graphsignal.com/docs/integrations/openai/) and [LangChain](https://graphsignal.com/docs/integrations/langchain/). Traces, errors, and data, such as prompts and completions, are automatically recorded and available for analysis at [app.graphsignal.com](https://app.graphsignal.com/).

Refer to the guides below for detailed information on:

* [Manual Tracing](https://graphsignal.com/docs/guides/manual-tracing/)
* [Session Tracking](https://graphsignal.com/docs/guides/session-tracking/)
* [Cost and Usage Monitoring](https://graphsignal.com/docs/guides/cost-and-usage-monitoring/)
* [Scores and Feedback](https://graphsignal.com/docs/guides/scores-and-feedback/)

See [API reference](https://graphsignal.com/docs/reference/python-api/) for full documentation.

Some integration examples are available in [examples](https://github.com/graphsignal/examples) repo.


## Analyze

[Log in](https://app.graphsignal.com/) to Graphsignal to monitor and analyze your application.


## Overhead

Graphsignal tracer is very lightweight. The overhead per trace is measured to be less than 100 microseconds.


## Security and Privacy

Graphsignal tracer can only open outbound connections to `api.graphsignal.com` and send data, no inbound connections or commands are possible.

Payloads, such as prompts and completions, are recorded by default in case of automatic tracing. To disable, set `record_payloads=False` in `graphsignal.configure`.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://api.graphsignal.com` are allowed.

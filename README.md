# Graphsignal: Observability for AI Stack

[![License](http://img.shields.io/github/license/graphsignal/graphsignal-python)](https://github.com/graphsignal/graphsignal-python/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal-python?label=version)](https://github.com/graphsignal/graphsignal-python)
[![Status](https://img.shields.io/uptimerobot/status/m787882560-d6b932eb0068e8e4ade7f40c?label=SaaS%20status)](https://stats.uptimerobot.com/gMBNpCqqqJ)


Graphsignal is an observability platform for AI agents and LLM-powered applications. It helps developers ensure AI applications run as expected and users have the best experience. With Graphsignal, developers can:

* Trace generations, runs, and sessions with full AI context.
* Score any user interactions and application execution.
* See latency breakdowns and distributions.
* Analyze model API costs for deployments, models, or users.
* Get notified about errors and anomalies.
* Monitor API, compute, and GPU utilization.

[![Dashboards](https://graphsignal.com/external/screencast-dashboards.gif)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).


## Install

Install Graphsignal library by running:

```
pip install --upgrade graphsignal
```

Or clone and install the [GitHub repository](https://github.com/graphsignal/graphsignal-python):

```
git clone https://github.com/graphsignal/graphsignal-python.git
python setup.py install
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

### Automatic integration

Graphsignal **auto-instruments** and traces libraries and frameworks, such as [OpenAI](https://graphsignal.com/docs/integrations/openai/) and [LangChain](https://graphsignal.com/docs/integrations/langchain/). Traces, errors, and data, such as prompts and completions, are automatically recorded and available for analysis at [app.graphsignal.com](https://app.graphsignal.com/).

Some integration examples are available in [examples](https://github.com/graphsignal/examples) repo.


## Session tracking

Session groups multiple traces together to represent a run, thread, conversation or user interactions. Session tracking allows session-level visualization, analytics and issue detection. It also enables detection of session outliers and other issues.

Set a session identifier as `session_id` tag for every request, e.g. in a request handler:

```python
graphsignal.set_context_tag('session_id', session_id)
```

or directly, when tracing manually:

```python
with graphsignal.trace(tags=dict(session_id=session_id)):
    ...
```

If you are running a single process per session and added Graphsignal at command line, you can set the `session_id` tag in an environment variable.

```sh
env GRAPHSIGNAL_TAGS="session_id=123" python -m graphsignal <script>
```

## Session tracking

Session groups multiple traces together to represent a run, thread, conversation or user interactions. Session tracking allows session-level visualization, analytics and issue detection. It also enables detection of session outliers and other issues.

Set a session identifier as `session_id` tag for every request, e.g. in a request handler:

```python
graphsignal.set_context_tag('session_id', session_id)
```

or directly, when tracing manually:

```python
with graphsignal.trace(tags=dict(session_id=session_id)):
    ...
```

If you are running a single process per session and added Graphsignal at command line, you can set the `session_id` tag in an environment variable.

```bash
env GRAPHSIGNAL_TAGS="session_id=123" python -m graphsignal <script>
```

### User tracking

User tracking allows grouping and visualization of user-related traces, interactions, metrics, and costs. It also enables detection of user interaction outliers and other events.

To enable user tracking, set user identifier as `user_id` tag for every request, e.g. in a request handler:

```python
graphsignal.set_context_tag('user_id', user_id)
```

or directly, when tracing manually:

```python
with graphsignal.trace(tags=dict(user_id=user_id)):
    ...
```

If you are running a single process per user and added Graphsignal at command line, you can set the `user_id` tag in an environment variable.

```bash
env GRAPHSIGNAL_TAGS="user_id=123" python -m graphsignal <script>
```


**Scores and feedback**

Scores allow recording an evaluation of any event or object, such as generation, run, session, or user. Scores can be associated with events or objects using tags, but can also be set directly to a span.

Tag request, run, session, or user for each request or run:

```python
graphsignal.set_context_tag('run_id', run_id)
```

or directly, when tracing manually:

```python
with graphsignal.trace('generate', tags=dict('run_id', run_id)):
    ...
```

Create a score for a tag. This can be done at a later time and/or by other application. For example, when user clicks thumbs-up or thumbs-down for a request or a session:

```python
graphsignal.score('user_feedback', tags=dict('run_id', run_id), score=1, comment=user_comment)
```

You can also associate a score with a span directly:

```python
with graphsignal.trace('generate') as span:
    ...
    span.score('prompt_injection', score=0.7, severity=2)

```

See API reference for more information on [`graphsignal.score`](https://graphsignal.com/docs/reference/python-api/#graphsignalscore) and [`Span.score`](https://graphsignal.com/docs/reference/python-api/#graphsignalspanscore) methods.


**Manual tracing**

To measure and monitor operations that are not automatically instrumented, wrap the code with [`trace()`](https://graphsignal.com/docs/reference/python-api/#graphsignaltrace) method or use [`@trace_function`](https://graphsignal.com/docs/reference/python-api/#graphsignaltrace_function) decorator.

To record payloads and track usage metrics, use [`Span.set_payload()`](https://graphsignal.com/docs/reference/python-api/#graphsignalspanset_payload). 

```python
with graphsignal.trace('my-operation') as span:
    ...
    span.set_payload('my-data', data, usage=dict(size=my_data_size))
```

```python
@graphsignal.trace_function
def my_function():
    ...
```

When tracing LLM generations, provide payloads in [OpenAI format](https://platform.openai.com/docs/api-reference/chat), which is supported by Graphsignal. Set `model_type='chat'` tag and add input and output data as `input` and `output` payloads respectively.

```python
with graphsignal.trace('generate', tags=dict(model_type='chat')) as span:
    output_data = my_llm_call(input_data)
    ...
    span.set_payload('input', input_data, usage=dict(token_count=input_token_count))
    span.set_payload('output', input_data, usage=dict(token_count=output_token_count))
```

For auto-instrumented libraries, or when using `@trace_function` decorator, `trace()` method with `with` context manager or callbacks, exceptions are **automatically** recorded. For other cases, use [`Span.add_exception`](https://graphsignal.com/docs/reference/python-api/#graphsignalspanadd_exception).


## Analyze

[Log in](https://app.graphsignal.com/) to Graphsignal to monitor and analyze your application and monitor for issues.


## Overhead

Graphsignal tracer is very lightweight. The overhead per trace is measured to be less than 100 microseconds.


## Security and Privacy

Graphsignal tracer can only open outbound connections to `signal-api.graphsignal.com` and send data, no inbound connections or commands are possible.

Payloads, such as prompts and completions, are recorded by default in case of automatic tracing. To disable, set `record_payloads=False` in `graphsignal.configure`.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://signal-api.graphsignal.com` are allowed.

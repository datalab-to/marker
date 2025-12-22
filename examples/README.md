## Usage Examples

This directory contains examples of running `marker` in different contexts.

### Usage with Modal

We have a [self-contained example](./marker_modal_deployment.py) that shows how you can quickly use [Modal](https://modal.com) to deploy `marker` by provisioning a container with a GPU, and expose that with an API so you can submit PDFs for conversion into Markdown, HTML, or JSON.

It's a limited example that you can extend into different use cases.

#### Pre-requisites

Make sure you have the `modal` client installed by [following their instructions here](https://modal.com/docs/guide#getting-started).

Modal's [Starter Plan](https://modal.com/pricing) includes $30 of free compute each month.
Modal is [serverless](https://arxiv.org/abs/1902.03383), so you only pay for resources when you are using them.

#### Running the example

Once `modal` is configured, you can deploy it to your workspace by running:

> modal deploy marker_modal_deployment.py

Notes:
- `marker` has a few models it uses. By default, the endpoint will check if these models are loaded and download them if not (first request will be slow). You can avoid this by running

> modal run marker_modal_deployment.py::download_models

Which will create a [`Modal Volume`](https://modal.com/docs/guide/Volumes) to store them for reuse.

Once the deploy is finished, you can:
- Test a file upload locally through your CLI using an `invoke_conversion` command we expose through Modal's [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint)
- Get the URL of your endpoint and make a request through a client of your choice.

**Test from your CLI with `invoke_conversion`**

If your endpoint is live, simply run this command:

```
$ modal run marker_modal_deployment.py::invoke_conversion --pdf-file <PDF_FILE_PATH> --output-format markdown
```

And it'll automatically detect the URL of your new endpoint using [`.get_web_url()`](https://modal.com/docs/guide/webhook-urls#determine-the-url-of-a-web-endpoint-from-code), make sure it's healthy, submit your file, and store its output on your machine (in the same directory).

**Making a request using your own client**

If you want to make requests elsewhere e.g. with cURL or a client like Insomnia, you'll need to get the URL.

When your `modal deploy` command from earlier finishes, it'll include your endpoint URL at the end. For example:

```
$ modal deploy marker_modal_deployment.py
...
✓ Created objects.
├── 🔨 Created mount /marker/examples/marker_modal_deployment.py
├── 🔨 Created function download_models.
├── 🔨 Created function MarkerModalDemoService.*.
└── 🔨 Created web endpoint for MarkerModalDemoService.fastapi_app => <YOUR_ENDPOINT_URL>
✓ App deployed in 149.877s! 🎉
```

If you accidentally close your terminal session, you can also always go into Modal's dashboard and:
  - Find the app (default name: `datalab-marker-modal-demo`)
  - Click on `MarkerModalDemoService`
  - Find your endpoint URL

Once you have your URL, make a request to `{YOUR_ENDPOINT_URL}/convert` like this (you can also use Insomnia, etc.):
```
curl --request POST \
  --url {BASE_URL}/convert \
  --header 'Content-Type: multipart/form-data' \
  --form file=@/Users/cooldev/sample.pdf \
  --form output_format=html
  ```

You should get a response like this

```
{
	"success": true,
	"filename": "sample.pdf",
	"output_format": "html",
	"json": null,
	"html": "<YOUR_RESPONSE_CONTENT>",
	"markdown": null,
	"images": {},
	"metadata": {... page level metadata ...},
	"page_count": 2
}
```

[Modal](https://modal.com) makes deploying and scaling models and inference workloads much easier.

If you're interested in Datalab's managed API or on-prem document intelligence solution, check out [our platform here](https://datalab.to/?utm_source=gh-marker).

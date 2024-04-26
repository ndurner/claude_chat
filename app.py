import gradio as gr
import base64
import os
from anthropic import Anthropic
import json

from doc2json import process_docx
from settings_mgr import generate_download_settings_js, generate_upload_settings_js

dump_controls = False
log_to_console = False

# constants
image_embed_prefix = "🖼️🆙 "

def encode_image(image_data):
    """Generates a prefix for image base64 data in the required format for the
    four known image formats: png, jpeg, gif, and webp.

    Args:
    image_data: The image data, encoded in base64.

    Returns:
    An object encoding the image
    """

    # Get the first few bytes of the image data.
    magic_number = image_data[:4]
  
    # Check the magic number to determine the image type.
    if magic_number.startswith(b'\x89PNG'):
        image_type = 'png'
    elif magic_number.startswith(b'\xFF\xD8'):
        image_type = 'jpeg'
    elif magic_number.startswith(b'GIF89a'):
        image_type = 'gif'
    elif magic_number.startswith(b'RIFF'):
        if image_data[8:12] == b'WEBP':
            image_type = 'webp'
        else:
            # Unknown image type.
            raise Exception("Unknown image type")
    else:
        # Unknown image type.
        raise Exception("Unknown image type")

    return {"type": "base64",
            "media_type": "image/" + image_type,
            "data": base64.b64encode(image_data).decode('utf-8')}

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, files):
    for file in files:
        if file.name.endswith(".docx"):
            content = process_docx(file.name)
        else:
            with open(file.name, mode="rb") as f:
                content = f.read()

                if isinstance(content, bytes):
                    content = content.decode('utf-8', 'replace')
                else:
                    content = str(content)

        fn = os.path.basename(file.name)
        history = history + [(f'```{fn}\n{content}\n```', None)]

    return history

def add_img(history, files):
    for file in files:
        if log_to_console:
            print(f"add_img {file.name}")
        history = history + [(image_embed_prefix + file.name, None)]

        gr.Info(f"Image added as {file.name}")

    return history

def submit_text(txt_value):
    return add_text([chatbot, txt_value], [chatbot, txt_value])

def undo(history):
    history.pop()
    return history

def dump(history):
    return str(history)

def load_settings():  
    # Dummy Python function, actual loading is done in JS  
    pass  

def save_settings(acc, sec, prompt, temp, tokens, model):  
    # Dummy Python function, actual saving is done in JS  
    pass  

def process_values_js():
    return """
    () => {
        return ["api_key", "system_prompt"];
    }
    """

def bot(message, history, api_key, system_prompt, temperature, max_tokens, model):
    try:
        client = Anthropic(
            api_key=api_key
        )

        if log_to_console:
            print(f"bot history: {str(history)}")

        history_openai_format = []
        user_msg_parts = []
        for human, assi in history:
            if human is not None:
                if human.startswith(image_embed_prefix):
                    with open(human.lstrip(image_embed_prefix), mode="rb") as f:
                        content = f.read()
                    user_msg_parts.append({"type": "image",
                                           "source": encode_image(content)})
                else:
                    user_msg_parts.append({"type": "text", "text": human})

            if assi is not None:
                if user_msg_parts:
                    history_openai_format.append({"role": "user", "content": user_msg_parts})
                    user_msg_parts = []

                history_openai_format.append({"role": "assistant", "content": assi})

        if message:
            user_msg_parts.append({"type": "text", "text": human})
        
        if user_msg_parts:
            history_openai_format.append({"role": "user", "content": user_msg_parts})

        if log_to_console:
            print(f"br_prompt: {str(history_openai_format)}")

        response = client.messages.create(
            model=model,
            messages= history_openai_format,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_prompt
        )

        if log_to_console:
            print(f"br_response: {str(response)}")

        resp = ""
        for content in response.content:
            resp += content.text

        history[-1][1] = resp
        if log_to_console:
            print(f"br_result: {str(history)}")

    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

    return "", history

def import_history(history, file):
    with open(file.name, mode="rb") as f:
        content = f.read()

        if isinstance(content, bytes):
            content = content.decode('utf-8', 'replace')
        else:
            content = str(content)

    # Deserialize the JSON content
    import_data = json.loads(content)

    new_history = import_data.get('history', history)  # Use existing history if not provided
    new_system_prompt = import_data.get('system_prompt', '')  # Default to empty if not provided

    return new_history, new_system_prompt

with gr.Blocks() as demo:
    gr.Markdown("# Anthropic™️ Claude™️ Chat (Nils' Version™️)")

    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/oai_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (Anthropic) and hosting provider (Hugging Face).""")
        
        api_key = gr.Textbox(label="Anthropic API Key", elem_id="api_key")
        model = gr.Dropdown(label="Model", value="claude-3-opus-20240229", allow_custom_value=True, elem_id="model",
                            choices=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1", "claude-2.0", "claude-instant-1.2"])
        system_prompt = gr.TextArea("You are a helpful yet diligent AI assistant. Answer faithfully and factually correct. Respond with 'I do not know' if uncertain.", label="System Prompt", lines=3, max_lines=250, elem_id="system_prompt")  
        temp = gr.Slider(0, 1, label="Temperature", elem_id="temp", value=1)
        max_tokens = gr.Slider(1, 4000, label="Max. Tokens", elem_id="max_tokens", value=800)
        save_button = gr.Button("Save Settings")  
        load_button = gr.Button("Load Settings")  
        dl_settings_button = gr.Button("Download Settings")
        ul_settings_button = gr.Button("Upload Settings")

        load_button.click(load_settings, js="""  
            () => {  
                let elems = ['#api_key textarea', '#system_prompt textarea', '#temp input', '#max_tokens input', '#model'];
                elems.forEach(elem => {
                    let item = document.querySelector(elem);
                    let event = new InputEvent('input', { bubbles: true });
                    item.value = localStorage.getItem(elem.split(" ")[0].slice(1)) || '';
                    item.dispatchEvent(event);
                });
            }  
        """)

        save_button.click(save_settings, [api_key, system_prompt, temp, max_tokens, model], js="""  
            (oai, sys, temp, ntok, model) => {  
                localStorage.setItem('api_key', oai);  
                localStorage.setItem('system_prompt', sys);  
                localStorage.setItem('temp', document.querySelector('#temp input').value);  
                localStorage.setItem('max_tokens', document.querySelector('#max_tokens input').value);  
                localStorage.setItem('model', model);  
            }  
        """) 

        control_ids = [('api_key', '#api_key textarea'),
                       ('system_prompt', '#system_prompt textarea'),
                       ('temp', '#temp input'),
                       ('max_tokens', '#max_tokens input'),
                       ('model', '#model')]
        controls = [api_key, system_prompt, temp, max_tokens, model]

        dl_settings_button.click(None, controls, js=generate_download_settings_js("claude_chat_settings.bin", control_ids))
        ul_settings_button.click(None, None, None, js=generate_upload_settings_js(control_ids))

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        show_copy_button=True,
        height=350
    )

    with gr.Row():
        btn = gr.UploadButton("📁 Upload", size="sm", file_count="multiple")
        img_btn = gr.UploadButton("🖼️ Upload", size="sm", file_count="multiple", file_types=["image"])
        undo_btn = gr.Button("↩️ Undo")
        undo_btn.click(undo, inputs=[chatbot], outputs=[chatbot])

        clear = gr.ClearButton(chatbot, value="🗑️ Clear")

    with gr.Row():
        txt = gr.TextArea(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload a file",
            container=False,
            lines=3,            
        )
        submit_btn = gr.Button("🚀 Send", scale=0)
        submit_click = submit_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [txt, chatbot, api_key, system_prompt, temp, max_tokens, model], [txt, chatbot],
        )
        submit_click.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    if dump_controls:
        with gr.Row():
            dmp_btn = gr.Button("Dump")
            txt_dmp = gr.Textbox("Dump")
            dmp_btn.click(dump, inputs=[chatbot], outputs=[txt_dmp])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [txt, chatbot, api_key, system_prompt, temp, max_tokens, model], [txt, chatbot],
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False, postprocess=False)
    img_msg = img_btn.upload(add_img, [chatbot, img_btn], [chatbot], queue=False, postprocess=False)

    with gr.Accordion("Import/Export", open = False):
        import_button = gr.UploadButton("History Import")
        export_button = gr.Button("History Export")
        export_button.click(lambda: None, [chatbot, system_prompt], js="""
            (chat_history, system_prompt) => {
                const export_data = {
                    history: chat_history,
                    system_prompt: system_prompt
                };
                const history_json = JSON.stringify(export_data);
                const blob = new Blob([history_json], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'chat_history.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            """)
        dl_button = gr.Button("File download")
        dl_button.click(lambda: None, [chatbot], js="""
            (chat_history) => {
                // Attempt to extract content enclosed in backticks with an optional filename
                const contentRegex = /```(\\S*\\.(\\S+))?\\n?([\\s\\S]*?)```/;
                const match = contentRegex.exec(chat_history[chat_history.length - 1][1]);
                if (match && match[3]) {
                    // Extract the content and the file extension
                    const content = match[3];
                    const fileExtension = match[2] || 'txt'; // Default to .txt if extension is not found
                    const filename = match[1] || `download.${fileExtension}`;
                    // Create a Blob from the content
                    const blob = new Blob([content], {type: `text/${fileExtension}`});
                    // Create a download link for the Blob
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    // If the filename from the chat history doesn't have an extension, append the default
                    a.download = filename.includes('.') ? filename : `${filename}.${fileExtension}`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                } else {
                    // Inform the user if the content is malformed or missing
                    alert('Sorry, the file content could not be found or is in an unrecognized format.');
                }
            }
        """)
        import_button.upload(import_history, inputs=[chatbot, import_button], outputs=[chatbot, system_prompt])

demo.queue().launch()
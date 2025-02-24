# from huggingface_hub import HfApi
# from huggingface_hub.commands.user import _login

# _login(HfApi(), token="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")
from huggingface_hub._login import _login
_login(token='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', add_to_git_credential=False)
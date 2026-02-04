"""SD Tokenizer for text embedding."""
import logging
import os
import traceback
import torch
from transformers import CLIPTokenizerFast


def model_options_long_clip(sd, tokenizer_data, model_options):
    """Handle long CLIP models."""
    return tokenizer_data, model_options


def parse_parentheses(string):
    """Parse nested parentheses into list."""
    result, current, level = [], "", 0
    for char in string:
        if char == "(":
            if level == 0 and current:
                result.append(current)
                current = "("
            elif level == 0:
                current = "("
            else:
                current += char
            level += 1
        elif char == ")":
            level -= 1
            if level == 0:
                result.append(current + ")")
                current = ""
            else:
                current += char
        else:
            current += char
    if current:
        result.append(current)
    return result


def token_weights(string, weight=1.0):
    """Parse string into tokens with weights."""
    out = []
    for x in parse_parentheses(string):
        w = weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x, w = x[1:-1], weight * 1.1
            if (xx := x.rfind(":")) > 0:
                try:
                    w, x = float(x[xx + 1:]), x[:xx]
                except ValueError:
                    pass
            out += token_weights(x, w)
        else:
            out.append((x, weight))
    return out


def escape_important(text):
    return text.replace("\\)", "\0\1").replace("\\(", "\0\2")


def unescape_important(text):
    return text.replace("\0\1", ")").replace("\0\2", "(")


def expand_directory_list(directories):
    """Expand directories to include subdirectories."""
    dirs = set(directories)
    for x in directories:
        for root, _, _ in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)


def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    """Load embedding from directory."""
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]
    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except Exception:
            continue
        
        if os.path.isfile(embed_path):
            valid_file = embed_path
        else:
            for ext in [".safetensors", ".pt", ".bin"]:
                if os.path.isfile(embed_path + ext):
                    valid_file = embed_path + ext
                    break
        if valid_file:
            break

    if not valid_file:
        return None

    try:
        if valid_file.lower().endswith(".safetensors"):
            import safetensors.torch
            embed = safetensors.torch.load_file(valid_file, device="cpu")
        else:
            embed = torch.load(valid_file, weights_only=True, map_location="cpu")
    except Exception:
        logging.warning(f"{traceback.format_exc()}\n\nerror loading embedding: {embedding_name}")
        return None

    if "string_to_param" in embed:
        return next(iter(embed["string_to_param"].values()))
    if isinstance(embed, list):
        out_list = [t.reshape(-1, t.shape[-1]) for x in embed for k, t in x.items() if t.shape[-1] == embedding_size]
        return torch.cat(out_list, dim=0) if out_list else None
    if embed_key and embed_key in embed:
        return embed[embed_key]
    return next(iter(embed.values()))


class SDTokenizer:
    """Stable Diffusion tokenizer."""
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True,
                 embedding_directory=None, embedding_size=768, embedding_key="clip_l",
                 tokenizer_class=CLIPTokenizerFast, has_start_token=True,
                 pad_to_max_length=True, min_length=None):
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path or "include/sd1_tokenizer/")
        self.max_length = max_length
        self.min_length = min_length
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        self.embedding_directory = embedding_directory
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"

        empty = self.tokenizer("")["input_ids"]
        self.tokens_start = 1 if has_start_token else 0
        self.start_token = empty[0] if has_start_token else None
        self.end_token = empty[1] if has_start_token else empty[0]
        self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def _try_get_embedding(self, name):
        embed = load_embed(name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None and (stripped := name.strip(",")) != name:
            embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
            return embed, name[len(stripped):]
        return embed, ""

    def tokenize_with_weights(self, text, return_word_ids=False):
        pad_token = self.end_token if self.pad_with_end else 0
        parsed = token_weights(escape_important(text), 1.0)
        tokens = []

        for segment, weight in parsed:
            for word in unescape_important(segment).replace("\n", " ").split():
                if word.startswith(self.embedding_identifier) and self.embedding_directory:
                    name = word[len(self.embedding_identifier):].strip("\n")
                    embed, leftover = self._try_get_embedding(name)
                    if embed is None:
                        logging.warning(f"embedding:{name} does not exist")
                    else:
                        tokens.append([(embed[i], weight) for i in range(embed.shape[0])] if len(embed.shape) > 1 else [(embed, weight)])
                        print("loading", name)
                    if leftover:
                        word = leftover
                    else:
                        continue
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])

        batched = []
        batch = [(self.start_token, 1.0, 0)] if self.start_token else []
        batched.append(batch)

        for i, t_group in enumerate(tokens):
            is_large = len(t_group) >= self.max_word_length
            while t_group:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining = self.max_length - len(batch) - 1
                    if is_large:
                        batch.extend([(t, w, i + 1) for t, w in t_group[:remaining]])
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining:]
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * remaining)
                    batch = [(self.start_token, 1.0, 0)] if self.start_token else []
                    batched.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        return batched if return_word_ids else [[(t, w) for t, w, _ in x] for x in batched]

    def untokenize(self, pairs):
        return [(a, self.inv_vocab[a[0]]) for a in pairs]


class SD1Tokenizer:
    """SD1 Tokenizer wrapper."""
    def __init__(self, embedding_directory=None, clip_name="l", tokenizer=SDTokenizer):
        self.clip_name = clip_name
        self.clip = f"clip_{clip_name}"
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory))

    def tokenize_with_weights(self, text, return_word_ids=False):
        return {self.clip_name: getattr(self, self.clip).tokenize_with_weights(text, return_word_ids)}

    def untokenize(self, pairs):
        return getattr(self, self.clip).untokenize(pairs)

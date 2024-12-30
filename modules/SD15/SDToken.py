import logging
import os
import traceback
import torch
from transformers import CLIPTokenizer


def parse_parentheses(string: str) -> list:
    """#### Parse a string with nested parentheses.

    #### Args:
        - `string` (str): The input string.

    #### Returns:
        - `list`: The parsed list of strings.
    """
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string: str, current_weight: float) -> list:
    """#### Parse a string into tokens with weights.

    #### Args:
        - `string` (str): The input string.
        - `current_weight` (float): The current weight.

    #### Returns:
        - `list`: The list of token-weight pairs.
    """
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text: str) -> str:
    """#### Escape important characters in a string.

    #### Args:
        - `text` (str): The input text.

    #### Returns:
        - `str`: The escaped text.
    """
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text: str) -> str:
    """#### Unescape important characters in a string.

    #### Args:
        - `text` (str): The input text.

    #### Returns:
        - `str`: The unescaped text.
    """
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def expand_directory_list(directories: list) -> list:
    """#### Expand a list of directories to include all subdirectories.

    #### Args:
        - `directories` (list): The list of directories.

    #### Returns:
        - `list`: The expanded list of directories.
    """
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)


def load_embed(embedding_name: str, embedding_directory: list, embedding_size: int, embed_key: str = None) -> torch.Tensor:
    """#### Load an embedding from a directory.

    #### Args:
        - `embedding_name` (str): The name of the embedding.
        - `embedding_directory` (list): The list of directories to search.
        - `embedding_size` (int): The size of the embedding.
        - `embed_key` (str, optional): The key for the embedding. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The loaded embedding.
    """
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
        except:
            continue
        if not os.path.isfile(embed_path):
            extensions = [".safetensors", ".pt", ".bin"]
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch

            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            if "weights_only" in torch.load.__code__.co_varnames:
                embed = torch.load(embed_path, weights_only=True, map_location="cpu")
            else:
                embed = torch.load(embed_path, map_location="cpu")
    except Exception:
        logging.warning(
            "{}\n\nerror loading embedding, skipping loading: {}".format(
                traceback.format_exc(), embedding_name
            )
        )
        return None

    if embed_out is None:
        if "string_to_param" in embed:
            values = embed["string_to_param"].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            values = embed.values()
            embed_out = next(iter(values))
    return embed_out


class SDTokenizer:
    """#### Class representing a Stable Diffusion tokenizer."""

    def __init__(
        self,
        tokenizer_path: str = None,
        max_length: int = 77,
        pad_with_end: bool = True,
        embedding_directory: str = None,
        embedding_size: int = 768,
        embedding_key: str = "clip_l",
        tokenizer_class: type = CLIPTokenizer,
        has_start_token: bool = True,
        pad_to_max_length: bool = True,
        min_length: int = None,
    ):
        """#### Initialize the SDTokenizer.

        #### Args:
            - `tokenizer_path` (str, optional): The path to the tokenizer. Defaults to None.
            - `max_length` (int, optional): The maximum length of the input. Defaults to 77.
            - `pad_with_end` (bool, optional): Whether to pad with the end token. Defaults to True.
            - `embedding_directory` (str, optional): The directory for embeddings. Defaults to None.
            - `embedding_size` (int, optional): The size of the embeddings. Defaults to 768.
            - `embedding_key` (str, optional): The key for the embeddings. Defaults to "clip_l".
            - `tokenizer_class` (type, optional): The tokenizer class. Defaults to CLIPTokenizer.
            - `has_start_token` (bool, optional): Whether the tokenizer has a start token. Defaults to True.
            - `pad_to_max_length` (bool, optional): Whether to pad to the maximum length. Defaults to True.
            - `min_length` (int, optional): The minimum length of the input. Defaults to None.
        """
        if tokenizer_path is None:
            tokenizer_path = "_internal/sd1_tokenizer/"
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name: str) -> tuple:
        """#### Try to get an embedding.

        #### Args:
            - `embedding_name` (str): The name of the embedding.

        #### Returns:
            - `tuple`: The embedding and any leftover text.
        """
        embed = load_embed(
            embedding_name,
            self.embedding_directory,
            self.embedding_size,
            self.embedding_key,
        )
        if embed is None:
            stripped = embedding_name.strip(",")
            if len(stripped) < len(embedding_name):
                embed = load_embed(
                    stripped,
                    self.embedding_directory,
                    self.embedding_size,
                    self.embedding_key,
                )
                return (embed, embedding_name[len(stripped) :])
        return (embed, "")

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> list:
        """#### Tokenize text with weights.

        #### Args:
            - `text` (str): The input text.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `list`: The tokenized text with weights.
        """
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = (
                unescape_important(weighted_segment).replace("\n", " ").split(" ")
            )
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if (
                    word.startswith(self.embedding_identifier)
                    and self.embedding_directory is not None
                ):
                    embedding_name = word[len(self.embedding_identifier) :].strip("\n")
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(
                            f"warning, embedding:{embedding_name} does not exist, ignoring"
                        )
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append(
                                [(embed[x], weight) for x in range(embed.shape[0])]
                            )
                        print("loading ", embedding_name)
                    # if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                # parse word
                tokens.append(
                    [
                        (t, weight)
                        for t in self.tokenizer(word)["input_ids"][
                            self.tokens_start : -1
                        ]
                    ]
                )

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    # break word in two and add end token
                    if is_large:
                        batch.extend(
                            [(t, w, i + 1) for t, w in t_group[:remaining_length]]
                        )
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens

    def untokenize(self, token_weight_pair: list) -> list:
        """#### Untokenize a list of token-weight pairs.

        #### Args:
            - `token_weight_pair` (list): The list of token-weight pairs.

        #### Returns:
            - `list`: The untokenized list.
        """
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))


class SD1Tokenizer:
    """#### Class representing the SD1Tokenizer."""

    def __init__(self, embedding_directory: str = None, clip_name: str = "l", tokenizer: type = SDTokenizer):
        """#### Initialize the SD1Tokenizer.

        #### Args:
            - `embedding_directory` (str, optional): The directory for embeddings. Defaults to None.
            - `clip_name` (str, optional): The name of the CLIP model. Defaults to "l".
            - `tokenizer` (type, optional): The tokenizer class. Defaults to SDTokenizer.
        """
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory))

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict:
        """#### Tokenize text with weights.

        #### Args:
            - `text` (str): The input text.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `dict`: The tokenized text with weights.
        """
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(
            text, return_word_ids
        )
        return out

    def untokenize(self, token_weight_pair: list) -> list:
        """#### Untokenize a list of token-weight pairs.

        #### Args:
            - `token_weight_pair` (list): The list of token-weight pairs.

        #### Returns:
            - `list`: The untokenized list.
        """
        return getattr(self, self.clip).untokenize(token_weight_pair)
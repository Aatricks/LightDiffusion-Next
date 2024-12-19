import ollama

from modules.Utilities import util


def enhance_prompt(p=None):
    prompt, neg, width, height, cfg = util.load_parameters_from_file()
    if p is None:
        pass
    else:
        prompt = p
    print(prompt)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"""Your goal is to generate a text-to-image prompt based on a user's input, detailing their desired final outcome for an image. The user will provide specific details about the characteristics, features, or elements they want the image to include. The prompt should guide the generation of an image that aligns with the user's desired outcome.

                        Generate a text-to-image prompt by arranging the following blocks in a single string, separated by commas:

                        Image Type: [Specify desired image type]

                        Aesthetic or Mood: [Describe desired aesthetic or mood]

                        Lighting Conditions: [Specify desired lighting conditions]

                        Composition or Framing: [Provide details about desired composition or framing]

                        Background: [Specify desired background elements or setting]

                        Colors: [Mention any specific colors or color palette]

                        Objects or Elements: [List specific objects or features]

                        Style or Artistic Influence: [Mention desired artistic style or influence]

                        Subject's Appearance: [Describe appearance of main subject]

                        Ensure the blocks are arranged in order of visual importance, from the most significant to the least significant, to effectively guide image generation, a block can be surrounded by parentheses to gain additionnal significance.

                        This is an example of a user's input: "a beautiful blonde lady in lingerie sitting in seiza in a seducing way with a focus on her assets"

                        And this is an example of a desired output: "portrait| serene and mysterious| soft, diffused lighting| close-up shot, emphasizing facial features| simple and blurred background| earthy tones with a hint of warm highlights| renaissance painting| a beautiful lady with freckles and dark makeup"
                        
                        Here is the user's input: {prompt}

                        Write the prompt in the same style as the example above, in a single line , with absolutely no additional information, words or symbols other than the enhanced prompt.

                        Output:""",
            },
        ],
    )
    print("here's the enhanced prompt :", response["message"]["content"])
    return response["message"]["content"]



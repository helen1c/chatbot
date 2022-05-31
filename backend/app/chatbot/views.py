from rest_framework.response import Response
from rest_framework.views import APIView
from .generator import Generator
from .generator_utils import available_generators
import random


default_args = {
    "max_history": 5,
    "sliding_window": True,
    "sliding_window_size": 5,
    "sampling": True,
    "max_seq_len": 128,
    "fp16": False,
    "temperature": 0.9,
    "top_k": -1,
    "top_p": 0.9,
}

model_arg_name = "model_name_or_path"
checkpoint_arg_name = "init_checkpoint"

generator_dct = {"generator": None}


class GenerateDialogView(APIView):

    permission_classes = ()

    def post(self, request, *args, **kwargs):

        data = request.data
        if not data:
            return Response({"message": "data not found, missing user prompt"}, 404)

        if generator_dct["generator"] is None:
            return Response({"message": "model is not loaded"}, 404)

        user_input = data["user"]

        bot_generated = generator_dct["generator"].generate(user_input)

        return Response({"bot_prompt": f"{bot_generated}"}, 200)


class ChooseModelView(APIView):

    permission_classes = ()

    def post(self, request, *args, **kwargs):

        data = request.data
        if not data:
            return Response({"message": "data not found, missing user prompt"}, 404)

        bot_personality = data["personality"]

        if not bot_personality:
            return Response({"message": f"personality field must be given"}, 404)

        personality_dict = available_generators[bot_personality]

        if not personality_dict:
            return Response(
                {"message": f"personality: {bot_personality} does not exist"}, 404
            )

        r_seed = random.randint(1, 2 * 20)

        default_args[model_arg_name] = personality_dict[model_arg_name]
        default_args[checkpoint_arg_name] = personality_dict[checkpoint_arg_name]
        default_args["seed"] = r_seed

        generator_dct["generator"] = Generator(default_args)

        print("Model successfully loaded.")

        return Response(200)

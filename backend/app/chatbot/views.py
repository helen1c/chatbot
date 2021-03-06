from rest_framework.response import Response
from rest_framework.views import APIView
from .generator import Generator
from .generator_utils import available_generators, load_generator_models
import random
import uuid

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
    "device": "cpu",
    "n_gpu": 0,
}


model_arg_name = "model_name_or_path"
checkpoint_arg_name = "init_checkpoint"

available_models = load_generator_models(default_args, 3)

generator_dct = {}


class GetAvailableModelsView(APIView):

    permission_classes = ()

    def get(self, request, *args, **kwargs):
        available_models = []
        loaded = 0
        for key in available_generators.keys():
            available_models.append(
                {"model_id": key, "model_name": available_generators[key]["model_name"]}
            )
            loaded += 1
            if loaded >= 3:
                break

        print(available_models)
        return Response({"available_models": available_models}, 200)


class GenerateDialogView(APIView):

    permission_classes = ()

    def post(self, request, *args, **kwargs):

        data = request.data
        if not data:
            return Response({"message": "data not found, missing user prompt"}, 404)

        specified_uuid = data["specified_uuid"]
        if specified_uuid is None or generator_dct.get(specified_uuid) is None:
            return Response(
                {
                    "message": "It seems that your model is not initialized. Please go back and choose model again."
                },
                404,
            )

        print(generator_dct.keys())

        user_input = data["user"]
        bot_generated = generator_dct[specified_uuid].generate(user_input)

        return Response({"bot_prompt": f"{bot_generated}"}, 200)


class ChooseModelView(APIView):

    permission_classes = ()

    def post(self, request, *args, **kwargs):

        data = request.data
        if not data:
            return Response({"message": "data not found, missing user prompt"}, 404)

        print(data)
        bot_personality_id = data["personality"]

        if bot_personality_id is None:
            return Response({"message": f"personality field must be given"}, 404)

        personality_dict = available_models[bot_personality_id]

        if personality_dict is None:
            return Response(
                {"message": f"personality: {bot_personality_id} does not exist"}, 404
            )

        r_seed = random.randint(1, 2 ** 25)

        # default_args[model_arg_name] = personality_dict[model_arg_name]
        # default_args[checkpoint_arg_name] = personality_dict[checkpoint_arg_name]
        default_args["seed"] = r_seed

        specified_uuid_ex = data.get("specified_uuid")

        if (
            specified_uuid_ex is not None
            and generator_dct.get(specified_uuid_ex) is not None
        ):
            specified_uuid = specified_uuid_ex
        else:
            specified_uuid = str(uuid.uuid4())

        generator_dct[specified_uuid] = Generator(personality_dict, default_args)

        print("Generator successfully created.")
        return Response({"specified_uuid": specified_uuid}, 200)

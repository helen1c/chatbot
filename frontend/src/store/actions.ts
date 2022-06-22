import { createAction } from '@reduxjs/toolkit';
import { ActionTypes, Model } from '../types';

function withPayloadType<T>() {
  return (t: T) => ({ payload: t });
}

export const sendMessage = createAction(
  ActionTypes.SEND_MESSAGE,
  withPayloadType<{ message: string }>(),
);

export const sendMessageSuccess = createAction(
  ActionTypes.SEND_MESSAGE_SUCCESS,
  withPayloadType<{ reply: string }>(),
);

export const sendMessageFailure = createAction(
  ActionTypes.SEND_MESSAGE_FAILURE,
  withPayloadType<{ error: any }>(),
);

export const selectModel = createAction(
  ActionTypes.SELECT_MODEL,
  withPayloadType<{ model: Model }>(),
);

export const getModels = createAction(ActionTypes.GET_MODELS);
export const getModelsSuccess = createAction(
  ActionTypes.GET_MODELS_SUCCESS,
  withPayloadType<{ models: Model[] }>(),
);
export const getModelsFailure = createAction(
  ActionTypes.GET_MODELS_FAILURE,
  withPayloadType<{ error: any }>(),
);

export const loadModel = createAction(
  ActionTypes.LOAD_MODEL,
  withPayloadType<{ model: Model }>(),
);

export const loadModelSuccess = createAction(
  ActionTypes.LOAD_MODEL_SUCCESS,
  withPayloadType<{ userUid: string }>(),
);

export const loadModelFailure = createAction(
  ActionTypes.LOAD_MODEL_FAILURE,
  withPayloadType<{ error: any }>(),
);

export const unsetReply = createAction(ActionTypes.UNSET_REPLY);

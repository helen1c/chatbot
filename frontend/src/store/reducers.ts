import { createReducer, PayloadAction } from '@reduxjs/toolkit';
import { ModelSate } from '../types';
import {
  getModels,
  getModelsFailure,
  getModelsSuccess,
  loadModel,
  loadModelFailure,
  loadModelSuccess,
  selectModel,
  sendMessage,
  sendMessageFailure,
  sendMessageSuccess,
  unsetReply,
} from './actions';

const initialState: ModelSate = {
  userUuid: '',
  error: undefined,
  loading: false,
  models: [],
  selectedModel: undefined,
  reply: '',
  replying: false,
};

export const modelReducer = createReducer(initialState, {
  [unsetReply.type]: state => ({ ...state, reply: '', replying: false }),
  [sendMessage.type]: state => ({ ...state, reply: '', replying: true }),
  [sendMessageSuccess.type]: (
    state,
    action: PayloadAction<ReturnType<typeof sendMessageSuccess>['payload']>,
  ) => ({ ...state, reply: action.payload.reply, replying: false }),

  [sendMessageFailure.type]: (
    state,
    action: PayloadAction<ReturnType<typeof sendMessageFailure>['payload']>,
  ) => ({ ...state, error: action.payload.error, reply: '', replying: false }),

  [selectModel.type]: (
    state,
    action: PayloadAction<ReturnType<typeof selectModel>['payload']>,
  ) => ({ ...state, selectedModel: action.payload.model }),

  [getModels.type]: state => ({
    ...state,
    loading: true,
    models: [],
  }),
  [getModelsSuccess.type]: (
    state,
    action: PayloadAction<ReturnType<typeof getModelsSuccess>['payload']>,
  ) => ({
    ...state,
    loading: false,
    models: action.payload.models,
  }),
  [getModelsFailure.type]: (
    state,
    action: PayloadAction<ReturnType<typeof getModelsFailure>['payload']>,
  ) => ({
    ...state,
    loading: false,
    error: action.payload.error,
  }),

  [loadModel.type]: (
    state,
    action: PayloadAction<ReturnType<typeof loadModel>['payload']>,
  ) => ({
    ...state,
    loading: true,
    selectedModel: action.payload.model,
  }),

  [loadModelSuccess.type]: (
    state,
    action: PayloadAction<ReturnType<typeof loadModelSuccess>['payload']>,
  ) => ({
    ...state,
    loading: false,
    userUuid: action.payload.userUid,
  }),

  [loadModelFailure.type]: (
    state,
    action: PayloadAction<ReturnType<typeof loadModelFailure>['payload']>,
  ) => ({
    ...state,
    loading: false,
    error: action.payload.error,
  }),
});

import { RouterState } from 'connected-react-router';

export enum Sender {
  BOT,
  HUMAN,
}

export interface Message {
  sender: Sender;
  message: string;
}

export interface Model {
  id: number;
  name: string;
}

export interface ModelSate {
  userUuid: string;
  selectedModel?: Model;
  loading: boolean;
  models: Model[];
  reply: string;
  error: any;
  replying: boolean;
}

export interface State {
  model: ModelSate;
  router: RouterState;
}

export enum ActionTypes {
  GET_MODELS = 'GET_MODELS',
  GET_MODELS_SUCCESS = 'GET_MODELS_SUCCESS',
  GET_MODELS_FAILURE = 'GET_MODELS_FAILURE',

  LOAD_MODEL = 'LOAD_MODEL',
  LOAD_MODEL_SUCCESS = 'LOAD_MODEL_SUCCESS',
  LOAD_MODEL_FAILURE = 'LOAD_MODEL_FAILURE',

  SELECT_MODEL = 'SELECT_MODEL',
  SEND_MESSAGE = 'SEND_MESSAGE',
  SEND_MESSAGE_SUCCESS = 'SEND_MESSAGE_SUCCESS',
  SEND_MESSAGE_FAILURE = 'SEND_MESSAGE_FAILURE',

  UNSET_REPLY = 'UNSET_REPLY',
}

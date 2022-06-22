import { all, call, takeEvery, put, select } from 'redux-saga/effects';
import { ActionTypes, Model } from '../types';
import {
  getModelsFailure,
  getModelsSuccess,
  loadModel,
  loadModelFailure,
  loadModelSuccess,
  sendMessage,
  sendMessageFailure,
  sendMessageSuccess,
} from './actions';
import { selectUserId } from './selectors';
import { httpRequest } from '../api/rest-client';

function* getModelsSaga() {
  try {
    const { data } = yield call(httpRequest, 'get', '/chatbot/get_available_models/');
    yield put(getModelsSuccess({ models: data.available_models.map((m: any): Model => ({ id: m.model_id, name: m.model_name })) }));
  } catch (e) {
    yield put(getModelsFailure({ error: e }));
  }
}

function* loadModelSaga({
  payload: { model },
}: ReturnType<typeof loadModel>) {
  try {
    const currentId: string = yield select(selectUserId);
    let body: any = { personality: model.id }
    if (currentId) {
      body.specified_uuid = currentId
    }
    console.log(body)
    const { data } = yield call(httpRequest, 'post', '/chatbot/choose_model/', { data: body });
    yield put(loadModelSuccess({ userUid: data.specified_uuid || currentId }));
  } catch (e) {
    yield put(loadModelFailure({ error: e }));
  }
}

function* sendMessageSaga({
  payload: { message },
}: ReturnType<typeof sendMessage>) {
  try {
    const currentId: string = yield select(selectUserId);
    const body = { user: message, specified_uuid: currentId };
    const { data } = yield call(httpRequest, 'post', '/chatbot/generate_dialog/', { data: body });
    yield put(
      sendMessageSuccess({ reply: data.bot_prompt }),
    );
  } catch (e) {
    yield put(sendMessageFailure({ error: e }));
  }
}

function* appSagas() {
  yield all([
    takeEvery(ActionTypes.GET_MODELS, getModelsSaga),
    takeEvery(ActionTypes.LOAD_MODEL, loadModelSaga),
    takeEvery(ActionTypes.SEND_MESSAGE, sendMessageSaga),
  ]);
}

export function* sagas() {
  while (true) {
    yield all([call(appSagas)]);
  }
}

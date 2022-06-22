import { connectRouter, routerMiddleware } from 'connected-react-router';
import { createBrowserHistory, History } from 'history';
import {
  applyMiddleware,
  combineReducers,
  createStore,
  Middleware,
  compose,
} from '@reduxjs/toolkit';
import createSagaMiddleware, { SagaMiddleware } from 'redux-saga';
import { sagas } from './sagas';
import { State } from '../types';
import { modelReducer } from './reducers';

function createRootReducer(history: History) {
  return combineReducers<State>({
    model: modelReducer,
    router: connectRouter(history),
  });
}

const history: History = createBrowserHistory({
  //@ts-ignore
  basename: process.env.PUBLIC_URL,
});

const router: Middleware = routerMiddleware(history);
const rootReducer = createRootReducer(history);
const sagaMiddleware: SagaMiddleware = createSagaMiddleware();
const enhancer = compose(applyMiddleware(...[sagaMiddleware, router]));

const configureStore = (initialState = {}) => {
  return createStore(rootReducer, initialState, enhancer);
};

const runSaga = () => sagaMiddleware.run(sagas);

export { configureStore, history, runSaga };

import React from 'react';
import App from './app/App';
import reportWebVitals from './reportWebVitals';
import { configureStore, runSaga, history } from './store/create-store';
import { ConnectedRouter } from 'connected-react-router';
import { registerResponseInterceptors } from './api/rest-client';
import { errorInterceptor, responseInterceptor } from './api/interceptors';
import { Provider } from 'react-redux';
import { createRoot } from 'react-dom/client';
import 'rsuite/dist/rsuite.min.css';
import './index.css';

interface IProps {
  Component: typeof App;
}

const store = configureStore();
registerResponseInterceptors(
  responseInterceptor(store),
  errorInterceptor(store),
);
runSaga();

const MOUNT_NODE = document.getElementById('root') as HTMLElement;

const ConnectedApp: React.FC<IProps> = ({ Component }) => (
  <Provider store={store}>
    {/*@ts-ignore*/}
    <ConnectedRouter history={history}>
      <Component />
    </ConnectedRouter>
  </Provider>
);

const render = (Component: typeof App) => {
  const root = createRoot(MOUNT_NODE);
  root.render(<ConnectedApp Component={Component} />);
};

render(App);
reportWebVitals();

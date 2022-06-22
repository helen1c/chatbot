import axios, { AxiosRequestConfig, AxiosResponse, Method } from 'axios';

const instance = axios.create();
const REST_SERVICE_ROOT_PATH = process.env.REACT_APP_REST_SERVICE_ROOT_PATH;

export const registerResponseInterceptors = (
  responseInterceptor: (
    response: AxiosResponse<any>,
  ) => AxiosResponse<any> | Promise<AxiosResponse<any>>,
  errorInterceptor: (error: any) => () => void,
) => {
  instance.interceptors.response.use(responseInterceptor, errorInterceptor);
};

const getRequestObject = (
  method: Method,
  url: string,
  config: AxiosRequestConfig,
  accept: string,
): AxiosRequestConfig => ({
  ...config,
  headers: {
    Accept: accept,
    ...config.headers,
  },
  method,
  params: {
    ...config.params,
  },
  url,
});

const directHttpRequest = (
  method: Method,
  url: string,
  config = { params: {} } as AxiosRequestConfig,
  accept = 'application/json',
) => {
  const requestObject = getRequestObject(method, url, config, accept);
  return instance.request(requestObject);
};

export const httpRequest = (
  method: Method,
  url: string,
  config = { params: {} } as AxiosRequestConfig,
  accept = 'application/json',
) =>
  directHttpRequest(method, `${REST_SERVICE_ROOT_PATH}${url}`, config, accept);

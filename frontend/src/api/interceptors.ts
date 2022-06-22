import { AxiosResponse } from 'axios';
import { State } from '../types';
import { Store } from '@reduxjs/toolkit';

export const responseInterceptor = (store: Store<State>) => (
  response: AxiosResponse,
) => response;

export const errorInterceptor = (store: Store<State>) => (error: Error) => () =>
  console.log(error);

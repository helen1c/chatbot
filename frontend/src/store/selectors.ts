import { State } from '../types';

export const selectUserId = (state: State) => state.model.userUuid;
export const selectModels = (state: State) => state.model.models;
export const selectLoading = (state: State) => state.model.loading;
export const selectCurrentModel = (state: State) => state.model.selectedModel;
export const selectError = (state: State) => state.model.error;
export const selectReply = (state: State) => state.model.reply;
export const selectReplying = (state: State) => state.model.replying;

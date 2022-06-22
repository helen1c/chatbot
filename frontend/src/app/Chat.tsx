import React, { useEffect, useState } from 'react';
import { IconButton, Input, Loader, Panel } from 'rsuite';
import styled from 'styled-components';
import ReplyingIcon from '@rsuite/icons/More';
import SendIcon from '@rsuite/icons/Send';
import RobotIcon from '@rsuite/icons/Wechat';
import { Message, Model, Sender } from '../types';
import { DARK, LIGHT, LIGHTER, TEXT } from '../utils/colors';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectCurrentModel,
  selectLoading,
  selectReply,
  selectReplying,
} from '../store/selectors';
import { sendMessage, unsetReply } from '../store/actions';

const ChatContainer = styled(Panel)`
  flex-grow: 1;
  height: 100%;

  .rs-panel-body {
    display: flex;
    flex-direction: column;
    padding: 20px !important;
  }

  background: ${LIGHTER};
`;

const MessagesContainer = styled.div`
  flex-grow: 1;
  margin-bottom: 10px;
  border-radius: 6px;
  padding: 20px;
  overflow-y: auto;

  background: ${LIGHT};

  display: flex;
  flex-direction: column-reverse;
`;

const MessageBubble = styled.div<{ sender: Sender }>`
  align-self: ${({ sender }) => (sender === Sender.BOT ? 'start' : 'end')};
  background: ${({ sender }) => (sender === Sender.HUMAN ? TEXT : DARK)};
  color: ${({ sender }) =>
    sender === Sender.HUMAN ? '#272c36' : TEXT} !important;
  max-width: 50%;

  @media (max-width: 976px) {
    max-width: 95%;
  }

  margin-top: 12px;
  padding: 10px 20px 13px 20px;
  border-radius: 25px;
  white-space: break-spaces;
  word-wrap: break-word;
  color: ${TEXT};
  font-size: 20px;
`;

const MessageInputContainer = styled.div`
  display: flex;
`;

const MessageInput = styled(Input)`
  flex-grow: 1;
  width: auto;
  height: 50px;
  margin-right: 5px;
`;

const SendButton = styled(IconButton)`
  width: 50px;
  height: 50px;
  background: #fff;
`;

const Container = styled(Panel)`
  height: 100%;
  width: 100%;

  .rs-panel-body {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  background: ${LIGHTER};
`;

const Prompt = styled(Panel)`
  color: ${TEXT};
  background: ${LIGHTER};
  font-size: 30px;

  height: fit-content;
  background: ${LIGHT};
`;

const Icon = styled(RobotIcon)`
  margin: 0 10px;
`;

const Replying = styled(ReplyingIcon)`
  animation: beat 0.3s infinite alternate;
`;

const Chat: React.FC = () => {
  const dispatch = useDispatch();

  const selectedModel = useSelector(selectCurrentModel);
  const reply = useSelector(selectReply);
  const replying = useSelector(selectReplying);
  const loading = useSelector(selectLoading);

  const [message, setMessage] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [contextModel, setContextModel] = useState<Model | undefined>(
    selectedModel,
  );

  useEffect(() => {
    if (!contextModel || !selectedModel || contextModel.id !== selectedModel.id) {
      dispatch(unsetReply());
      setMessages([]);
      setContextModel(selectedModel);
      setMessage("");
    }
  }, [contextModel, dispatch, selectedModel]);

  useEffect(() => {
    if (reply) {
      setMessages([
        {
          sender: Sender.BOT,
          message: reply,
        },
        ...messages,
      ]);
      dispatch(unsetReply());
    }
    //eslint-disable-next-line
  }, [reply]);

  const handleSend = () => {
    if (message) {
      setMessages([
        {
          sender: Sender.HUMAN,
          message,
        },
        ...messages,
      ]);
      setMessage('');
      dispatch(sendMessage({ message }));
    }
  };

  return (
    <>
      {!!selectedModel && !loading && (
        <ChatContainer shaded>
          <MessagesContainer>
            {replying && (
              <MessageBubble sender={Sender.BOT}>
                <Replying />
              </MessageBubble>
            )}
            {messages.map((m, idx) => (
              <MessageBubble key={idx} sender={m.sender}>
                {m.message}
              </MessageBubble>
            ))}
          </MessagesContainer>
          <MessageInputContainer onKeyDown={e => e.keyCode === 13 && handleSend()}>
            <MessageInput
              value={message}
              onChange={(value: string) => setMessage(value)}
              disabled={replying}
            />
            <SendButton
              icon={<SendIcon width={30} height={30} />}
              onClick={() => handleSend()}
            />
          </MessageInputContainer>
        </ChatContainer>
      )}
      {!selectedModel && !loading && (
        <Container shaded>
          <Prompt shaded>
            ChatBot <Icon /> with personalities
          </Prompt>
        </Container>
      )}
      {loading && (
        <Container shaded>
          <Loader size={'lg'} inverse />
        </Container>
      )}
    </>
  );
};

export default Chat;

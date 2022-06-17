import React from 'react';
import styled from 'styled-components';
import { Panel } from 'rsuite';
import Menu from './Menu';
import Chat from './Chat';
import { DARK, LIGHT } from '../utils/colors';

const ContentWrapper = styled.div`
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 20px;
  background: ${DARK};
`;

const Content = styled(Panel)`
  position: relative;
  max-width: 1500px;
  width: 100%;
  height: 100%;

  background: ${LIGHT};

  .rs-panel-body {
    display: flex;
    height: 100%;

    @media (max-width: 976px) {
      padding: 0;
      box-shadow: none;
    }
  }
`;

const ContentLayout: React.FC = () => {
  return (
    <ContentWrapper>
      <Content shaded>
        <Menu />
        <Chat />
      </Content>
    </ContentWrapper>
  );
};

export default ContentLayout;

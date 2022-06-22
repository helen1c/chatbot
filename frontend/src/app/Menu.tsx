import React, { useCallback, useState } from 'react';
import { Button, IconButton, Loader, Panel } from 'rsuite';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { selectCurrentModel, selectModels } from '../store/selectors';
import ModelIcon from '@rsuite/icons/Project';
import { selectModel, loadModel } from '../store/actions';
import BackIcon from '@rsuite/icons/Menu';
import { LIGHT, LIGHTER, TEXT } from '../utils/colors';

const MenuContainer = styled(Panel) <{ selected?: boolean }>`
  position: relative;
  opacity: 1;
  transition: all 0.3s ease-in;

  background: ${LIGHTER};

  @media (max-width: 2000px) {
    min-width: 350px;
    margin-right: 20px;
    height: 100%;
  }

  @media (max-width: 976px) {
    position: absolute;
    margin: 0;
    min-width: 0;
    width: 100%;
    height: 100%;
    z-index: 999;
    box-shadow: none;
    padding: 10px 16px 10px 16px;

    ${({ selected }) => selected && 'opacity: 0; z-index: -1;'}
  }

  .rs-panel-body {
    display: flex;
    flex-direction: column;
    overflow-y: auto !important;

    @media (max-width: 976px) {
      &::-webkit-scrollbar {
        display: none;
      }
    }
  }
`;

const MenuItem = styled(Button)`
  width: 100%;
  height: 80px;
  margin-bottom: 10px;
  display: flex;
  justify-content: center;
  align-items: center;

  background: ${LIGHT};
  ${({ active }) => active && 'background: #fff !important;'}
  color: ${TEXT};
  font-size: 20px;
`;

const Icon = styled(ModelIcon)`
  margin-right: 5px;
`;

const BackButton = styled(IconButton) <{ selected?: boolean }>`
  @media (max-width: 2000px) {
    display: none;
  }

  @media (max-width: 976px) {
    display: initial;
    position: absolute;
    width: 50px;
    height: 50px;
    z-index: 999;
    border-radius: 6px;

    background: #fff;
  }
`;

const Menu: React.FC = () => {
  const dispatch = useDispatch();

  const models = useSelector(selectModels);
  const model = useSelector(selectCurrentModel);

  const [selected, setSelected] = useState<boolean>(!!model);

  const getMenuItems = useCallback(() => {
    return models.map(m => (
      <MenuItem
        key={m.id}
        onClick={() => {
          if (!model || m.id !== model.id) {
            dispatch(selectModel({ model: m }));
            dispatch(loadModel({ model: m }));
          }
          setSelected(true);
        }}
        active={model && m.id === model.id}
      >
        <Icon width={25} height={25} />
        {m.name}
      </MenuItem>
    ));
  }, [dispatch, model, models]);

  return (
    <>
      <BackButton
        type={'submit'}
        icon={<BackIcon width={30} height={30} />}
        onClick={() => setSelected(false)}
      />
      <MenuContainer shaded selected={selected}>
        {models.length === 0 && <Loader inverse size={'lg'} center />}
        {models.length > 0 && <div>{getMenuItems()}</div>}
      </MenuContainer>
    </>
  );
};

export default Menu;

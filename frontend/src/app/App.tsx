import React, { useEffect } from 'react';
import { Route, Switch } from 'react-router';
import { useDispatch, useSelector } from 'react-redux';
import { getModels } from '../store/actions';
import { selectModels } from '../store/selectors';
import ContentLayout from './ContentLayout';

const App: React.FC = () => {
  const dispatch = useDispatch();
  const models = useSelector(selectModels);
  useEffect(() => {
    dispatch(getModels());
  }, [dispatch]);

  return (
    <Switch>
      {models.map(m => (
        <Route
          exact
          path={`/chat-bot/${m.id}`}
          render={() => <ContentLayout />}
          key={m.id}
        />
      ))}
      <Route exact path={''} render={() => <ContentLayout />} />
    </Switch>
  );
};

export default App;

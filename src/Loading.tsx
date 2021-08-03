import React from 'react';

const Loading: React.FC<React.HTMLProps<HTMLDivElement>> = props => (
  <div {...props}>
    <h2>Loading...</h2>
  </div>
);

export default Loading;

import React from 'react';

import './ExpandCollapseButton.css';

const ChevronRight = () => (
  <svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' fill='#ccc' viewBox='0 0 16 16'>
    <path
      fillRule='evenodd'
      d='M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z'
    />
  </svg>
);

const ChevronDown = () => (
  <svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' fill='#ccc' viewBox='0 0 16 16'>
    <path
      fillRule='evenodd'
      d='M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z'
    />
  </svg>
);

interface ExpandCollapseButtonProps {
  isExpanded: boolean;
  setExpanded: (expanded: boolean) => void;
  style?: React.CSSProperties;
}

const ExpandCollapseButton: React.FC<ExpandCollapseButtonProps> = ({
  isExpanded,
  setExpanded,
  style,
}) => (
  <div
    style={style}
    role='button'
    className='expand-collapse-button'
    onClick={() => setExpanded(!isExpanded)}
  >
    {isExpanded ? <ChevronRight /> : <ChevronDown />}
  </div>
);

export default ExpandCollapseButton;

export const randomFloatInRange = (min: number, max: number): number =>
  Math.random() * (max - min) + min;

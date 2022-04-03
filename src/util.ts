export const randomFloatInRange = (min: number, max: number): number =>
  Math.random() * (max - min) + min;

// prettier-ignore
export const getHasSIMDSupport = () => WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]))

export const delay = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms));

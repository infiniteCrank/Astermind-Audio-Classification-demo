/// <reference types="vite/client" />

// Meyda has no bundled TS types in many builds
declare module 'meyda';

// Tell TS what a Vite `?worker` import returns
declare module '*?worker' {
    const WorkerFactory: new () => Worker;
    export default WorkerFactory;
}

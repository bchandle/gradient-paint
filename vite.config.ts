import { defineConfig } from 'vitest/config';
import { resolve } from 'node:path';

export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        gpuTests: resolve(import.meta.dirname, 'tests/gpu/index.html'),
      },
    },
  },
  test: {
    include: ['tests/unit/**/*.test.ts'],
    testTimeout: 60000,
  },
});

import fs from 'node:fs';

const css = fs.readFileSync(new URL('../src/styles/global.css', import.meta.url), 'utf8');
const packageJson = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const failures = [];

const frames = [
  ['min', 320],
  ['phone', 390],
  ['tablet', 834],
  ['desktop', 1440],
  ['max', 1920],
];
for (const [name, value] of frames) {
  if (!new RegExp(`--f-${name}:\\s*${value};`).test(css)) failures.push(`frame --f-${name} must equal ${value}`);
}
for (const [name, value] of [['md', 768], ['lg', 992]]) {
  if (!new RegExp(`--breakpoint-${name}:\\s*${value}px;`).test(css)) failures.push(`breakpoint ${name} must equal ${value}px`);
}

const roles = ['display', 'h1', 'h2', 'h3', 'h4', 'h5', 'para-1', 'para-2', 'eyebrow-1', 'cta'];
for (const role of roles) {
  const values = ['min', 'phone', 'tablet', 'desktop'].map((frame) => {
    const match = css.match(new RegExp(`--fs-${role}-${frame}:\\s*(\\d+(?:\\.\\d+)?);`));
    return match ? Number(match[1]) : null;
  });
  if (values.some((value) => value === null)) {
    failures.push(`type token ${role} is missing one or more frame sizes`);
  } else if (values.some((value, index) => index > 0 && value < values[index - 1])) {
    failures.push(`type token ${role} must increase from floor through desktop`);
  }
  if (!css.includes(`.text-${role} {`) || !css.includes(`font-size: var(--type-${role});`)) {
    failures.push(`type utility .text-${role} is missing`);
  }
}

const fontSizes = [...css.matchAll(/font-size:\s*([^;}]+)/g)].map((match) => match[1].trim());
if (fontSizes.some((value) => !value.startsWith('var(--type-') && value !== 'calc(16 * var(--px))')) {
  failures.push('font sizes must use type tokens (root rem scaling is the only exception)');
}
if (/@media\s*\([^)]*(?:max-width|min-width|width\s*[<>])/i.test(css)) {
  failures.push('layout must use the single desktop composition without width-specific layouts');
}
if (/\d+(?:\.\d+)?rem\b|\b100vh\b|\bmin-height:\s*100vh/.test(css)) {
  failures.push('design sizes must use --px; viewport-sized height must use svh/lvh/dvh');
}
if (!packageJson.scripts['check:scale']) failures.push('package.json must expose npm run check:scale');

if (failures.length) {
  console.error(failures.map((failure) => `✗ ${failure}`).join('\n'));
  process.exitCode = 1;
} else {
  console.log('Fluid scale verified: frame values, type tokens, desktop-only layout, and design units.');
}

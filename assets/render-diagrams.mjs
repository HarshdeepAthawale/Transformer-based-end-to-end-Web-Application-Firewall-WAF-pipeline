import mermaid from 'mermaid';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

mermaid.initialize({ startOnLoad: false });

const files = [
  { in: 'waf-architecture-diagram.mmd', out: '1-waf-architecture.svg' },
  { in: 'request-flow-ml-diagram.mmd', out: '2-request-flow-ml.svg' },
  { in: 'ml-pipeline-training-diagram.mmd', out: '3-ml-pipeline-training.svg' },
];

for (const { in: infile, out: outfile } of files) {
  const code = fs.readFileSync(path.join(__dirname, infile), 'utf8');
  const id = outfile.replace(/[^a-z0-9]/gi, '-');
  const { svg } = await mermaid.render(id, code);
  fs.writeFileSync(path.join(__dirname, outfile), svg);
  console.log('Written:', outfile);
}

console.log('Done. 3 SVGs in assets/');

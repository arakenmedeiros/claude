const { criarDeck } = require("./lib");
const path = require("path");
(async () => {
  const alvo = process.argv.slice(2);
  const todos = ["01","02","03","04","05","06","07","08","09","10","11","12","13"];
  for (const n of (alvo.length ? alvo : todos)) {
    const f = path.join(__dirname, "content", `ch${n}.js`);
    if (!require("fs").existsSync(f)) continue;
    const c = require(f);
    const arquivo = path.join(__dirname, "out", c.arq + ".pptx");
    const r = await criarDeck({ capNum: c.capNum, capTitulo: c.capTitulo, subtitulo: c.subtitulo, autor: "Araken de Medeiros Santos", arquivo, slides: c.slides });
    console.log(n, r);
  }
})();

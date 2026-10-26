const { montar } = require("./lib.js");
(async () => {
  const which = process.argv.slice(2).map(Number); const out = "/tmp/slides_work/out"; require("fs").mkdirSync(out, { recursive: true });
  const all = Object.assign({}, ...["c_a","c_b","c_c","c_d","c_e"].filter(f => require("fs").existsSync(`./${f}.js`)).map(f => require(`./${f}.js`)));
  for (const n of which.length ? which : Object.keys(all).map(Number)) { console.log(await montar(n, all[n], out)); }
})().catch(e => { console.error(e); process.exit(1); });

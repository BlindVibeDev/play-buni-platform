import { serve } from "bun";

serve({
  port: Number(process.env.PORT || 3000),
  fetch() {
    return new Response("Play Buni Frontend running with Bun\n");
  },
});

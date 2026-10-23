import http from "k6/http";
import { check } from "k6";
export const options = {
  scenarios: { health: { executor: "constant-vus", vus: 20, duration: "30s" } },
  thresholds: { http_req_failed: ["rate<0.01"], http_req_duration: ["p(95)<500"] }
};
export default function () {
  const response = http.get(__ENV.BASE_URL + "/health/live");
  check(response, { "liveness is healthy": (result) => result.status === 200 });
}

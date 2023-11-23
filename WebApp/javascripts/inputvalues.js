const value = document.querySelector("#RiceProdVal");
const input = document.querySelector("#RiceProduced");

value.textContent = input.value
input.addEventListener("input", (event) => {
    value.textContent = event.target.value;
  });


const calendarDates = document.getElementById("calendar-dates");
const monthYear = document.getElementById("month-year");
const prevMonth = document.getElementById("prev-month");
const nextMonth = document.getElementById("next-month");

const monthNames = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

let activeDate = new Date();
let selectedDateKey = null;

function renderCalendar() {
  const year = activeDate.getFullYear();
  const month = activeDate.getMonth();
  const firstDayOfMonth = new Date(year, month, 1);
  const lastDayOfMonth = new Date(year, month + 1, 0);
  const leadingDays = firstDayOfMonth.getDay();
  const totalDays = lastDayOfMonth.getDate();

  monthYear.textContent = `${monthNames[month]} ${year}`;
  calendarDates.innerHTML = "";

  for (let day = 0; day < leadingDays; day += 1) {
    const filler = buildDateCell("", ["is-muted"]);
    filler.disabled = true;
    calendarDates.appendChild(filler);
  }

  for (let dateNumber = 1; dateNumber <= totalDays; dateNumber += 1) {
    const date = new Date(year, month, dateNumber);
    const dateKey = date.toISOString().slice(0, 10);
    const modifiers = [];

    if (isToday(date)) {
      modifiers.push("is-today");
    }
    if (selectedDateKey === dateKey) {
      modifiers.push("is-selected");
    }

    const cell = buildDateCell(String(dateNumber), modifiers);
    cell.addEventListener("click", () => {
      selectedDateKey = dateKey;
      renderCalendar();
    });
    calendarDates.appendChild(cell);
  }
}

function buildDateCell(label, modifiers) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = ["date-cell", ...modifiers].join(" ").trim();
  button.textContent = label;
  return button;
}

function isToday(date) {
  const today = new Date();
  return (
    today.getFullYear() === date.getFullYear() &&
    today.getMonth() === date.getMonth() &&
    today.getDate() === date.getDate()
  );
}

prevMonth.addEventListener("click", () => {
  activeDate = new Date(activeDate.getFullYear(), activeDate.getMonth() - 1, 1);
  renderCalendar();
});

nextMonth.addEventListener("click", () => {
  activeDate = new Date(activeDate.getFullYear(), activeDate.getMonth() + 1, 1);
  renderCalendar();
});

renderCalendar();

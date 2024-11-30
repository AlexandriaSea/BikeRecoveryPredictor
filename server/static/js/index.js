let logisticRegressionRunCount = 0;
let randomForestRunCount = 0;

function isFormValid() {
  const form = document.getElementById("inputForm");

  if (form.checkValidity()) {
    // Proceed with the prediction if the form is valid
    console.log("Form is valid. Proceed with prediction...");
    return true;
    // Your prediction logic goes here
  } else {
    // Display an error message if the form is not valid
    form.reportValidity(); // This will show browser's built-in validation messages
    return false;
  }
}

function predictLogisticRegression() {
  if (!isFormValid()) {
    return;
  }

  const data = {
    PRIMARY_OFFENCE: document.getElementById("primary_offense_rf").value,
    OCC_YEAR: document.getElementById("occ_year_rf").value,
    OCC_MONTH: document.getElementById("occ_month_rf").value,
    OCC_DOW: document.getElementById('occ_dow_rf').value,
    OCC_DAY: document.getElementById("occ_day_rf").value,
    OCC_DOY: document.getElementById("occ_doy_rf").value,
    OCC_HOUR: document.getElementById("occ_hour_rf").value,
    REPORT_YEAR: document.getElementById("report_year_rf").value,
    REPORT_MONTH: document.getElementById("report_month_rf").value,
    REPORT_DOW: document.getElementById('report_dow_rf').value,
    REPORT_DAY: document.getElementById("report_day_rf").value,
    REPORT_DOY: document.getElementById("report_doy_rf").value,
    REPORT_HOUR: document.getElementById("report_hour_rf").value,
    DIVISION: document.getElementById("division_rf").value,
    LOCATION_TYPE: document.getElementById("location_type_rf").value,
    PREMISES_TYPE: document.getElementById("premises_type_rf").value,
    BIKE_MAKE: document.getElementById("bike_make_rf").value,
    BIKE_MODEL: document.getElementById("bike_model_rf").value,
    BIKE_TYPE: document.getElementById("bike_type_rf").value,
    BIKE_SPEED: document.getElementById("bike_speed_rf").value,
    BIKE_COLOUR: document.getElementById("bike_colour_rf").value,
    BIKE_COST: document.getElementById("bike_cost_rf").value,
    HOOD_158: document.getElementById("hood_158_rf").value,
    HOOD_140: document.getElementById("hood_140_rf").value,
  };

  fetch("/predict/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((result) => {
      // Remove placeholder if it exists
      const logisticHistory = document.getElementById("logisticHistory");
      if (logisticHistory.querySelector(".text-muted")) {
        logisticHistory.innerHTML = ""; // Clear the placeholder
      }

      // Append the result to the history list
      logisticRegressionRunCount++;
      const listItem = document.createElement("li");
      listItem.classList.add("list-group-item");
      listItem.textContent = `RUN ${logisticRegressionRunCount}: ${result.logistic_regression_prediction}`;
      logisticHistory.appendChild(listItem);
    })
    .catch((error) => console.error("Error:", error));
}

function predictRandomForest() {
  if (!isFormValid()) {
    return;
  }

  const data = {
    PRIMARY_OFFENCE: document.getElementById("primary_offense_rf").value,
    OCC_YEAR: document.getElementById("occ_year_rf").value,
    OCC_MONTH: document.getElementById("occ_month_rf").value,
    OCC_DOW: document.getElementById('occ_dow_rf').value,
    OCC_DAY: document.getElementById("occ_day_rf").value,
    OCC_DOY: document.getElementById("occ_doy_rf").value,
    OCC_HOUR: document.getElementById("occ_hour_rf").value,
    REPORT_YEAR: document.getElementById("report_year_rf").value,
    REPORT_MONTH: document.getElementById("report_month_rf").value,
    REPORT_DOW: document.getElementById('report_dow_rf').value,
    REPORT_DAY: document.getElementById("report_day_rf").value,
    REPORT_DOY: document.getElementById("report_doy_rf").value,
    REPORT_HOUR: document.getElementById("report_hour_rf").value,
    DIVISION: document.getElementById("division_rf").value,
    LOCATION_TYPE: document.getElementById("location_type_rf").value,
    PREMISES_TYPE: document.getElementById("premises_type_rf").value,
    BIKE_MAKE: document.getElementById("bike_make_rf").value,
    BIKE_MODEL: document.getElementById("bike_model_rf").value,
    BIKE_TYPE: document.getElementById("bike_type_rf").value,
    BIKE_SPEED: document.getElementById("bike_speed_rf").value,
    BIKE_COLOUR: document.getElementById("bike_colour_rf").value,
    BIKE_COST: document.getElementById("bike_cost_rf").value,
    HOOD_158: document.getElementById("hood_158_rf").value,
    HOOD_140: document.getElementById("hood_140_rf").value,
  };

  fetch("/predict/rf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((result) => {
      // Remove placeholder if it exists
      const randomForestHistory = document.getElementById(
        "randomForestHistory"
      );
      if (randomForestHistory.querySelector(".text-muted")) {
        randomForestHistory.innerHTML = ""; // Clear the placeholder
      }

      randomForestRunCount++;
      // Append the result to the history list
      const listItem = document.createElement("li");
      listItem.classList.add("list-group-item");
      listItem.textContent = `RUN ${randomForestRunCount}: ${result.random_forest_prediction}`;
      randomForestHistory.appendChild(listItem);
    })
    .catch((error) => console.error("Error:", error));
}

function clearForm() {
  // Reset all form fields to their default values
  document.getElementById("inputForm").reset();

  // Clear the displayed prediction results for both models
  document.getElementById("logisticHistory").innerHTML =
    "<li class='list-group-item text-muted'>Prediction results for Logistic Regression will appear here.</li>";
  document.getElementById("randomForestHistory").innerHTML =
    "<li class='list-group-item text-muted'>Prediction results for Random Forest will appear here.</li>";

  // Optionally, if you have other result containers or UI elements, you can clear them here as well
  console.log("Form has been cleared.");
}

// Change icon when the collapse state is toggled
document
  .querySelector('[data-bs-toggle="collapse"]')
  .addEventListener("click", function () {
    const icon = document.getElementById("collapseIcon");
    if (icon.classList.contains("bi-chevron-down")) {
      icon.classList.remove("bi-chevron-down");
      icon.classList.add("bi-chevron-up");
    } else {
      icon.classList.remove("bi-chevron-up");
      icon.classList.add("bi-chevron-down");
    }
  });

const fileInput = document.getElementById("project_zip");
const dropZone = document.getElementById("drop-zone");
const selectedFile = document.getElementById("selected-file");
const form = document.getElementById("analysis-form");
const button = document.getElementById("analyze-button");
const loadingBox = document.getElementById("loading-box");


// ==========================================================
// ZIP FILE SELECTION
// ==========================================================

if (fileInput) {

    fileInput.addEventListener("change", () => {

        if (
            fileInput.files.length > 0
            && selectedFile
        ) {

            selectedFile.textContent =
                fileInput.files[0].name;
        }

    });

}


// ==========================================================
// DRAG AND DROP ZIP
// ==========================================================

if (dropZone && fileInput) {

    [
        "dragenter",
        "dragover"
    ].forEach((eventName) => {

        dropZone.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                dropZone.classList.add(
                    "dragging"
                );

            }
        );

    });


    [
        "dragleave",
        "drop"
    ].forEach((eventName) => {

        dropZone.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                dropZone.classList.remove(
                    "dragging"
                );

            }
        );

    });


    dropZone.addEventListener(
        "drop",
        (event) => {

            const files =
                event.dataTransfer.files;

            if (files.length === 0) {
                return;
            }

            const file = files[0];


            // ----------------------------------------------
            // Only allow ZIP
            // ----------------------------------------------

            if (
                !file.name
                    .toLowerCase()
                    .endsWith(".zip")
            ) {

                if (selectedFile) {

                    selectedFile.textContent =
                        "Please choose a .zip file";

                }

                return;
            }


            // ----------------------------------------------
            // Put dropped ZIP into input
            // ----------------------------------------------

            const transfer =
                new DataTransfer();

            transfer.items.add(
                file
            );

            fileInput.files =
                transfer.files;


            if (selectedFile) {

                selectedFile.textContent =
                    file.name;

            }

        }
    );

}


// ==========================================================
// ANALYSIS LOADING MESSAGE
// ==========================================================

if (form) {

    form.addEventListener(
        "submit",
        () => {

            if (button) {

                button.disabled = true;

                button.textContent =
                    "Analyzing...";

            }


            if (loadingBox) {

                loadingBox.classList.remove(
                    "hidden"
                );

            }

        }
    );

}


// ==========================================================
// RESULTS PAGE SEARCH + FILTERS
// ==========================================================

const fileSearch =
    document.getElementById(
        "file-search"
    );

const riskFilter =
    document.getElementById(
        "risk-filter"
    );

const languageFilter =
    document.getElementById(
        "language-filter"
    );

const clearFilters =
    document.getElementById(
        "clear-filters"
    );

const visibleFileCount =
    document.getElementById(
        "visible-file-count"
    );

const emptyFilterState =
    document.getElementById(
        "empty-filter-state"
    );


const resultCards =
    Array.from(
        document.querySelectorAll(
            ".file-card"
        )
    );


// ==========================================================
// APPLY FILTERS
// ==========================================================

function applyResultFilters() {

    if (resultCards.length === 0) {
        return;
    }


    // ------------------------------------------------------
    // Search value
    // ------------------------------------------------------

    const searchValue =
        fileSearch
            ? fileSearch.value
                .trim()
                .toLowerCase()
            : "";


    // ------------------------------------------------------
    // Risk filter
    // ------------------------------------------------------

    const riskValue =
        riskFilter
            ? riskFilter.value
            : "all";


    // ------------------------------------------------------
    // Language filter
    // ------------------------------------------------------

    const languageValue =
        languageFilter
            ? languageFilter.value
            : "all";


    let visible = 0;


    // ------------------------------------------------------
    // Check every file card
    // ------------------------------------------------------

    resultCards.forEach(
        (card) => {

            const path =
                card.dataset.path || "";

            const risk =
                card.dataset.risk || "";

            const language =
                card.dataset.language || "";


            const matchesSearch = (
                !searchValue
                || path.includes(
                    searchValue
                )
            );


            const matchesRisk = (
                riskValue === "all"
                || risk === riskValue
            );


            const matchesLanguage = (
                languageValue === "all"
                || language === languageValue
            );


            const shouldShow = (
                matchesSearch
                && matchesRisk
                && matchesLanguage
            );


            card.classList.toggle(
                "filtered-out",
                !shouldShow
            );


            if (shouldShow) {

                visible += 1;

            }

        }
    );


    // ------------------------------------------------------
    // Update visible file count
    // ------------------------------------------------------

    if (visibleFileCount) {

        visibleFileCount.textContent =
            String(visible);

    }


    // ------------------------------------------------------
    // Show empty message when nothing matches
    // ------------------------------------------------------

    if (emptyFilterState) {

        emptyFilterState.classList.toggle(
            "hidden",
            visible !== 0
        );

    }

}


// ==========================================================
// SEARCH / FILTER EVENTS
// ==========================================================

[
    fileSearch,
    riskFilter,
    languageFilter
].forEach(
    (element) => {

        if (!element) {
            return;
        }


        element.addEventListener(
            "input",
            applyResultFilters
        );


        element.addEventListener(
            "change",
            applyResultFilters
        );

    }
);


// ==========================================================
// CLEAR FILTERS
// ==========================================================

if (clearFilters) {

    clearFilters.addEventListener(
        "click",
        () => {

            if (fileSearch) {

                fileSearch.value = "";

            }


            if (riskFilter) {

                riskFilter.value = "all";

            }


            if (languageFilter) {

                languageFilter.value = "all";

            }


            applyResultFilters();

        }
    );

}
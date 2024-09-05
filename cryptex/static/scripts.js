function identifyAlgorithm() {
    // Show the loader and loading text
    document.getElementById('loader').style.display = 'block';
    document.getElementById('loading-text').style.display = 'block';

    // Hide the result
    document.getElementById('result').style.display = 'none';

    // Get the ciphertext from the textarea
    const ciphertext = document.getElementById('ciphertext').value.trim();

    // Check if ciphertext is empty
    if (!ciphertext) {
        alert("Please enter some ciphertext before predicting.");
        document.getElementById('loader').style.display = 'none';
        document.getElementById('loading-text').style.display = 'none';
        return;
    }

    // Send a POST request to the Flask server with the ciphertext
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ciphertext: ciphertext })
    })
    .then(response => response.json())
    .then(data => {
        // Hide the loader and loading text
        document.getElementById('loader').style.display = 'none';
        document.getElementById('loading-text').style.display = 'none';

        // Display the result
        document.getElementById('result').style.display = 'block';
        document.getElementById('result').textContent = "Predicted Algorithm: " + data.algorithm;
    })
    .catch(error => {
        // Hide the loader and loading text
        document.getElementById('loader').style.display = 'none';
        document.getElementById('loading-text').style.display = 'none';

        // Display an error message
        document.getElementById('result').style.display = 'block';
        document.getElementById('result').textContent = "Error: Unable to predict the algorithm.";
        console.error('Error:', error);
    });
}

// Additional functions for clearing the input
function clearTextarea() {
    document.getElementById('ciphertext').value = ''; // Clear the textarea
    document.getElementById('result').textContent = ''; // Clear the result display
    document.getElementById('result').style.display = 'none'; // Hide the result
}

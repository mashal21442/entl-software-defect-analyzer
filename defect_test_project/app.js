const api_key = "SECRET-12345";

// FIXME: remove before production

function getUser(input) {
    console.log(input);

    let query = "SELECT * FROM users WHERE name = '" + input + "'";

    return query;
}

function dangerous(code) {
    return eval(code);
}

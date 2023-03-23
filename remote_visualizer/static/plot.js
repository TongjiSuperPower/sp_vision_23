async function sleep(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms)
    })
}

async function draw_plot() {
    let container = document.getElementById('container')
    let plot = document.getElementById('plot')

    let created = false
    let last_w = 0
    let last_h = 0
    let layout = {}
    let config = {}

    function update_plot(data_frames) {
        let trace_num = data_frames[0].values.length

        if (!created) {
            let data = []
            let names = data_frames[0].names
            for (let i = 0; i < trace_num; i++) {
                let trace = { y: [] }
                
                for (let j = 0; j < data_frames.length; j++) {
                    trace.y.push(data_frames[j].values[i])
                }
    
                if (names.length == trace_num) {
                    trace.name = names[i]
                }
                
                data.push(trace)
            }

            Plotly.newPlot(plot, data, layout, config)
            created = true
        }

        else {
            let traces = []
            let indices = []

            for (let i = 0; i < trace_num; i++) {
                let trace = []

                for (let j = 0; j < data_frames.length; j++) {
                    trace.push(data_frames[j].values[i])
                }            

                traces.push(trace)
                indices.push(i)
            }

            Plotly.extendTraces(plot, { y: traces }, indices)

            w = container.offsetWidth
            h = container.offsetHeight
            if (w != last_w || h != last_h) {
                Plotly.relayout(plot, { width: w, heigt: h })
                last_h = h
                last_w = w
            }
        }
    }

    while (true) {
        await fetch('/data')
            .then(response => response.json())
            .then((response) => {
                update_plot(response)
            })
        await sleep(10)
    }
}

document.addEventListener('DOMContentLoaded', function () {
    draw_plot()
})

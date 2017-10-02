(function($) {

    node_type_selects = $("select[name$='node_type']");
    node_type_selects.change(function() {
        select = $(this);
        select_fieldset = select.closest("fieldset");
        stochastic_fieldset = select_fieldset.nextAll("fieldset:has(h2:contains('Stochastic Type'))"); 
        deterministic_fieldset = select_fieldset.nextAll("fieldset:has(h2:contains('Deterministic Type'))");
        visualization_fieldset = select_fieldset.nextAll("fieldset:has(h2:contains('Visualization'))");

        if (select.val() == '0') { 
            // Stochastic Type
            deterministic_fieldset.hide(); 
            stochastic_fieldset.show();
            visualization_fieldset.show();
        } else { 
            // Deterministic Type
            stochastic_fieldset.hide();
            deterministic_fieldset.show();
            visualization_fieldset.hide();
        }
    });
    node_type_selects.change();

    is_observable_cbs = $("input[name$='is_observable']");
    is_observable_cbs.change(function() {
        io_cb = $(this);
        io_row = io_cb.closest("div[class$='is_observable']");
        ref_model_row = io_row.next(); 
        visualization_fieldset = io_cb.closest("fieldset").nextAll("fieldset:has(h2:contains('Visualization'))");

        if (io_cb.is(':checked')) { 
            ref_model_row.show();
            // Visualization is not supported for Observable Nodes 
            visualization_fieldset.hide();
        } else { 
            ref_model_row.hide();
            visualization_fieldset.show();
        }
    });
    is_observable_cbs.change();

    $(document).on('formset:added', function(event, $row, formsetName) {
        if (formsetName == 'nodes') {
            node_type_selects.change();
            is_observable_cbs.change();
        }
    });

    $(document).on('formset:removed', function(event, $row, formsetName) {
        // Row removed, nothing to do here yet
    });
    
})(django.jQuery);

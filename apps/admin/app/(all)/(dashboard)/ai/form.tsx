/**
 * Copyright (c) 2023-present Plane Software, Inc. and contributors
 * SPDX-License-Identifier: AGPL-3.0-only
 * See the LICENSE file for details.
 */

import { useMemo, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { Lightbulb, RefreshCw } from "lucide-react";
import { Button } from "@plane/propel/button";
import { TOAST_TYPE, setToast } from "@plane/propel/toast";
import { InstanceService } from "@plane/services";
import type { IFormattedInstanceConfiguration, TInstanceAIConfigurationKeys, TLLMProvider } from "@plane/types";
import { CustomSelect } from "@plane/ui";
// components
import { ControllerInput } from "@/components/common/controller-input";
// hooks
import { useInstance } from "@/hooks/store";

const instanceService = new InstanceService();

type IInstanceAIForm = {
  config: IFormattedInstanceConfiguration;
};

type AIFormValues = Record<TInstanceAIConfigurationKeys, string>;

const LLM_PROVIDER_OPTIONS: Record<TLLMProvider, { label: string; keyUrl: string; placeholder: string }> = {
  openai: { label: "OpenAI", keyUrl: "https://platform.openai.com/api-keys", placeholder: "sk-..." },
  gemini: { label: "Google Gemini", keyUrl: "https://aistudio.google.com/apikey", placeholder: "AIza..." },
  anthropic: { label: "Anthropic", keyUrl: "https://console.anthropic.com/settings/keys", placeholder: "sk-ant-..." },
};

const isProvider = (value: string): value is TLLMProvider => value in LLM_PROVIDER_OPTIONS;

export function InstanceAIForm(props: IInstanceAIForm) {
  const { config } = props;
  // store
  const { updateInstanceConfigurations } = useInstance();
  // states
  const [models, setModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  // form data
  const {
    handleSubmit,
    control,
    watch,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<AIFormValues>({
    defaultValues: {
      LLM_PROVIDER: isProvider(config["LLM_PROVIDER"] ?? "") ? config["LLM_PROVIDER"] : "openai",
      LLM_API_KEY: config["LLM_API_KEY"],
      LLM_MODEL: config["LLM_MODEL"],
    },
  });

  const provider = watch("LLM_PROVIDER");
  const apiKey = watch("LLM_API_KEY");
  const currentModel = watch("LLM_MODEL");
  const providerMeta = LLM_PROVIDER_OPTIONS[isProvider(provider) ? provider : "openai"];

  const modelOptions = useMemo(() => {
    const set = new Set(models);
    if (currentModel) set.add(currentModel);
    return Array.from(set);
  }, [models, currentModel]);

  const handleProviderChange = (value: TLLMProvider) => {
    setValue("LLM_PROVIDER", value, { shouldDirty: true });
    setModels([]);
  };

  const handleLoadModels = async () => {
    if (!isProvider(provider)) return;
    setIsLoadingModels(true);
    try {
      const response = await instanceService.listLLMModels({ provider, api_key: apiKey ?? "" });
      setModels(response.models);
      if (response.models.length === 0) {
        setToast({ type: TOAST_TYPE.WARNING, title: "No models", message: "The provider returned no chat models." });
      } else if (!response.models.includes(currentModel ?? "")) {
        setValue("LLM_MODEL", response.models[0], { shouldDirty: true });
      }
    } catch (err) {
      const message = (err as { error?: string } | undefined)?.error ?? "Could not load models. Check the API key.";
      setToast({ type: TOAST_TYPE.ERROR, title: "Error", message });
    } finally {
      setIsLoadingModels(false);
    }
  };

  const onSubmit = async (formData: AIFormValues) => {
    const payload: Partial<AIFormValues> = { ...formData };

    await updateInstanceConfigurations(payload)
      .then(() =>
        setToast({
          type: TOAST_TYPE.SUCCESS,
          title: "Success",
          message: "AI Settings updated successfully",
        })
      )
      .catch((err) => console.error(err));
  };

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div>
          <div className="pb-1 text-18 font-medium text-primary">AI provider</div>
          <div className="text-13 font-regular text-tertiary">
            Pick a provider, paste its API key, load the available models and choose one.
          </div>
        </div>
        <div className="grid-col grid w-full grid-cols-1 items-start justify-between gap-x-12 gap-y-8 lg:grid-cols-3">
          <div className="flex flex-col gap-1">
            <h4 className="text-13 text-tertiary">Provider</h4>
            <Controller
              control={control}
              name="LLM_PROVIDER"
              render={({ field: { value } }) => (
                <CustomSelect
                  value={value}
                  label={LLM_PROVIDER_OPTIONS[isProvider(value) ? value : "openai"].label}
                  onChange={handleProviderChange}
                  buttonClassName="rounded-md border-subtle"
                  input
                >
                  {(Object.keys(LLM_PROVIDER_OPTIONS) as TLLMProvider[]).map((key) => (
                    <CustomSelect.Option key={key} value={key} className="w-full">
                      {LLM_PROVIDER_OPTIONS[key].label}
                    </CustomSelect.Option>
                  ))}
                </CustomSelect>
              )}
            />
            <p className="pt-0.5 text-11 text-tertiary">Saved as LLM_PROVIDER.</p>
          </div>

          <ControllerInput
            control={control}
            type="password"
            name="LLM_API_KEY"
            label="API key"
            description={
              <>
                Create or copy your key{" "}
                <a
                  href={providerMeta.keyUrl}
                  target="_blank"
                  className="text-accent-primary hover:underline"
                  rel="noreferrer"
                  aria-label={`${providerMeta.label} API keys page`}
                >
                  here.
                </a>
              </>
            }
            placeholder={providerMeta.placeholder}
            error={Boolean(errors.LLM_API_KEY)}
            required={false}
          />

          <div className="flex flex-col gap-1">
            <h4 className="text-13 text-tertiary">Model</h4>
            <div className="flex items-center gap-2">
              <div className="grow">
                <Controller
                  control={control}
                  name="LLM_MODEL"
                  render={({ field: { value, onChange } }) => (
                    <CustomSelect
                      value={value}
                      label={value || "Load models to choose"}
                      onChange={onChange}
                      buttonClassName="rounded-md border-subtle"
                      disabled={modelOptions.length === 0}
                      maxHeight="lg"
                      input
                    >
                      {modelOptions.map((model) => (
                        <CustomSelect.Option key={model} value={model} className="w-full">
                          {model}
                        </CustomSelect.Option>
                      ))}
                    </CustomSelect>
                  )}
                />
              </div>
              <Button
                variant="secondary"
                size="lg"
                type="button"
                onClick={handleLoadModels}
                loading={isLoadingModels}
                disabled={isLoadingModels || !apiKey}
                aria-label="Load models"
              >
                <RefreshCw className="size-3.5" />
                {isLoadingModels ? "Loading" : "Load models"}
              </Button>
            </div>
            <p className="pt-0.5 text-11 text-tertiary">
              Models are fetched live from {providerMeta.label} using the API key above.
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-col items-start gap-4">
        <Button variant="primary" size="lg" onClick={handleSubmit(onSubmit)} loading={isSubmitting}>
          {isSubmitting ? "Saving" : "Save changes"}
        </Button>

        <div className="relative inline-flex items-center gap-1.5 rounded-sm border border-accent-subtle bg-accent-subtle px-4 py-2 text-caption-sm-regular text-accent-secondary">
          <Lightbulb className="size-4" />
          <div>Gemini and Anthropic are used through their OpenAI-compatible endpoints.</div>
        </div>
      </div>
    </div>
  );
}
